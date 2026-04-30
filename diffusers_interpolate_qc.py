"""
Standalone implementation of interpolate_qc using HuggingFace Diffusers.

Hierarchical bisection interpolation with CLIP-based quality control.
No dependencies on custom ControlNet implementation.

Author: Refactored from original cm.py
Date: 2024
"""

import os
import hashlib
import shutil
from typing import Optional, Tuple, List, Union

import torch
import numpy as np
from PIL import Image
from tqdm import trange

# HuggingFace Diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
)

# HuggingFace Transformers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPModel,
    CLIPProcessor,
)

# Torchvision for data augmentation
from torchvision import transforms


class DiffusersInterpolator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_type: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Auto-select dtype if not specified
        if dtype is None:
            if device == "cuda":
                dtype = torch.float16  # Use fp16 on CUDA for speed
            else:
                dtype = torch.float32  # MPS and CPU need fp32

        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        print(f"Using device: {device} with dtype: {dtype}")

        print(f"Loading Stable Diffusion from {model_id}...")

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
        self.vae.eval()

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
        self.unet.eval()

        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        self.controlnet = None
        # if controlnet_type:
        #     print(f"  Loading ControlNet ({controlnet_type})...")
        #     controlnet_id = self._get_controlnet_id(controlnet_type, model_id)
        #     self.controlnet = ControlNetModel.from_pretrained(
        #         controlnet_id,
        #         torch_dtype=dtype
        #     ).to(device)
        #     self.controlnet.eval()

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _get_controlnet_id(self, controlnet_type: str, base_model: str) -> str:
        """
        Get HuggingFace ControlNet model ID based on type and base model.

        Args:
            controlnet_type: "canny", "openpose", "depth", "seg", etc.
            base_model: Base SD model ID

        Returns:
            ControlNet model ID
        """
        # Map controlnet types to HuggingFace model IDs
        if "stable-diffusion-v1-5" in base_model or "v1-5" in base_model:
            controlnet_map = {
                "canny": "lllyasviel/sd-controlnet-canny",
                "openpose": "lllyasviel/sd-controlnet-openpose",
                "depth": "lllyasviel/sd-controlnet-depth",
                "seg": "lllyasviel/sd-controlnet-seg",
                "mlsd": "lllyasviel/sd-controlnet-mlsd",
                "normal": "lllyasviel/sd-controlnet-normal",
                "scribble": "lllyasviel/sd-controlnet-scribble",
            }
        elif "stable-diffusion-2" in base_model:
            # SD 2.x ControlNets (fewer available)
            controlnet_map = {
                "canny": "thibaud/controlnet-sd21-canny-diffusers",
                "depth": "thibaud/controlnet-sd21-depth-diffusers",
                "openpose": "thibaud/controlnet-sd21-openpose-diffusers",
            }
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        if controlnet_type not in controlnet_map:
            raise ValueError(
                f"Unknown controlnet type: {controlnet_type}. "
                f"Available: {list(controlnet_map.keys())}"
            )

        return controlnet_map[controlnet_type]

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        return text_embeddings

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image).astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) * 2.0
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)

        latent_dist = self.vae.encode(image_tensor).latent_dist
        latent = latent_dist.sample()

        latent = latent * self.vae.config.scaling_factor

        return latent

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        latent = latent / self.vae.config.scaling_factor

        image_tensor = self.vae.decode(latent).sample

        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)

        image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        image_np = (image_np * 255).round().astype(np.uint8)

        image = Image.fromarray(image_np[0])

        return image

    def _get_latent_stack(
        self,
        img1: Image.Image,
        img2: Image.Image,
        timestep_schedule: List[int],
        share_noise: bool = True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        L1 = self._encode_image(img1)
        L2 = self._encode_image(img2)

        latents1 = [L1]
        latents2 = [L2]

        for i in range(1, len(timestep_schedule)):
            t_prev = timestep_schedule[i - 1] if i > 0 else None
            t_now = timestep_schedule[i]

            noise = torch.randn_like(L1)

            latents1.append(self._add_noise(latents1[-1], noise, t_now, t_prev))

            if not share_noise:
                noise = torch.randn_like(L2)

            latents2.append(self._add_noise(latents2[-1], noise, t_now, t_prev))

        return latents1, latents2

    def _add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        t_now: int,
        t_prev: Optional[int] = None
    ) -> torch.Tensor:
        # Get alpha values from scheduler
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if t_prev is None:
            # Add noise from scratch: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
            sqrt_alpha = alphas_cumprod[t_now] ** 0.5
            sqrt_one_minus_alpha = (1 - alphas_cumprod[t_now]) ** 0.5

            noisy_latent = sqrt_alpha * latent + sqrt_one_minus_alpha * noise

        else:
            # Add incremental noise from t_prev to t_now
            # Given: x_{t_prev} = √ᾱ_{t_prev} * x_0 + √(1-ᾱ_{t_prev}) * ε_{prev}
            # Want:  x_{t_now} = √ᾱ_{t_now} * x_0 + √(1-ᾱ_{t_now}) * ε_{combined}
            # Where ε_{combined} includes ε_{prev} plus new noise

            a_prev = alphas_cumprod[t_prev] ** 0.5
            a_now = alphas_cumprod[t_now] ** 0.5
            sig_prev = (1 - alphas_cumprod[t_prev]) ** 0.5
            sig_now = (1 - alphas_cumprod[t_now]) ** 0.5

            # Scale factor for existing latent
            scale = a_now / a_prev

            # Additional noise needed
            sigma = (sig_now**2 - (scale * sig_prev)**2) ** 0.5

            noisy_latent = scale * latent + sigma * noise

        return noisy_latent

    @torch.no_grad()
    def _denoise_step(
        self,
        latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: int,
        guidance_scale: float = 7.5,
        controlnet_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Prepare timestep tensor
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)

        # Duplicate latent for CFG (uncond + cond)
        latent_model_input = torch.cat([latent, latent])

        # Scale latent for UNet (some schedulers require this)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # ControlNet forward pass (if enabled)
        down_block_res_samples = None
        mid_block_res_sample = None

        if self.controlnet is not None and controlnet_image is not None:
            # Duplicate control image for CFG
            controlnet_cond = torch.cat([controlnet_image, controlnet_image])

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

        # UNet forward pass
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # Scheduler step (denoise one step)
        latent = self.scheduler.step(
            noise_pred,
            timestep,
            latent,
            return_dict=False
        )[0]

        return latent

    def _optimize_embeddings(
        self,
        img1: Image.Image,
        img2: Image.Image,
        prompt: str,
        n_prompt: str,
        num_iters: int = 200,
        lr: float = 1e-4,
        guide_scale: float = 7.5,
        cache_path: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            state = torch.load(cache_path, map_location=self.device)
            return (
                state['cond1'].to(self.dtype),
                state['cond2'].to(self.dtype),
                state['uncond'].to(self.dtype),
            )

        print(f"Optimizing text embeddings ({num_iters} iterations)...")

        # Data augmentation (prevents overfitting)
        augment = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
        ])

        # Get base embeddings
        with torch.no_grad():
            cond_base = self._encode_prompt(prompt)
            uncond_base = self._encode_prompt(n_prompt)

        # Create learnable copies in fp32 so Adam doesn't overflow on fp16
        cond1 = cond_base.float().clone().detach().requires_grad_(True)
        cond2 = cond_base.float().clone().detach().requires_grad_(True)
        uncond = uncond_base.float().clone().detach().requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([cond1, cond2, uncond], lr=lr)

        # Setup scheduler for optimization
        self.scheduler.set_timesteps(50)  # Dummy schedule to get timesteps
        timesteps = self.scheduler.timesteps.cpu().numpy()

        # Training loop
        for iteration in range(num_iters):
            # Random timestep from middle range (not too easy, not too hard)
            t_idx = np.random.randint(len(timesteps) // 3, 2 * len(timesteps) // 3)
            t = int(timesteps[t_idx])

            # === Optimize embedding for img1 ===

            # Apply augmentation
            img1_aug = augment(img1)

            # Encode augmented image
            with torch.no_grad():
                L1 = self._encode_image(img1_aug)

            # Add noise at random timestep
            noise = torch.randn_like(L1)
            alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            sqrt_alpha = alphas_cumprod[t] ** 0.5
            sqrt_one_minus_alpha = (1 - alphas_cumprod[t]) ** 0.5
            noisy_L1 = sqrt_alpha * L1 + sqrt_one_minus_alpha * noise

            # Predict noise using current cond1 (cast to model dtype for UNet)
            text_emb = torch.cat([uncond, cond1]).to(self.dtype)
            latent_input = torch.cat([noisy_L1, noisy_L1])
            t_tensor = torch.tensor([t] * 2, device=self.device, dtype=torch.long)

            # UNet forward pass
            noise_pred = self.unet(
                latent_input,
                t_tensor,
                encoder_hidden_states=text_emb,
            ).sample

            # Apply CFG and compute loss in fp32 to avoid overflow
            noise_pred_uncond, noise_pred_cond = noise_pred.float().chunk(2)
            noise_pred_guided = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            loss1 = torch.nn.functional.mse_loss(noise_pred_guided, noise.float())

            # === Optimize embedding for img2 ===

            img2_aug = augment(img2)

            with torch.no_grad():
                L2 = self._encode_image(img2_aug)

            noise = torch.randn_like(L2)
            noisy_L2 = sqrt_alpha * L2 + sqrt_one_minus_alpha * noise

            text_emb = torch.cat([uncond, cond2]).to(self.dtype)
            latent_input = torch.cat([noisy_L2, noisy_L2])

            noise_pred = self.unet(
                latent_input,
                t_tensor,
                encoder_hidden_states=text_emb,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.float().chunk(2)
            noise_pred_guided = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            loss2 = torch.nn.functional.mse_loss(noise_pred_guided, noise.float())

            # Backprop and update
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log progress
            if iteration % 10 == 0:
                print(f"  Iter {iteration}/{num_iters}: "
                      f"loss1={loss1.item():.4f}, loss2={loss2.item():.4f}")

        print(f"✓ Optimization complete!")

        result = (cond1.detach().to(self.dtype), cond2.detach().to(self.dtype), uncond.detach().to(self.dtype))
        if cache_path:
            torch.save({'cond1': result[0], 'cond2': result[1], 'uncond': result[2]}, cache_path)
            print(f"Saved embeddings to {cache_path}")
        return result

    @torch.no_grad()
    def _evaluate_with_clip(
        self,
        image: Image.Image,
        pos_prompt: str,
        neg_prompt: str
    ) -> float:
        # Process inputs
        inputs = self.clip_processor(
            text=[pos_prompt, neg_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Compute similarities
        pos_sim = torch.nn.functional.cosine_similarity(
            image_embeds, text_embeds[0:1]
        )
        neg_sim = torch.nn.functional.cosine_similarity(
            image_embeds, text_embeds[1:2]
        )

        # Score = positive similarity - negative similarity
        score = (pos_sim - neg_sim).item()

        return score

    def interpolate_qc(
        self,
        img1: Image.Image,
        img2: Image.Image,
        prompt: str = "a photograph",
        n_prompt: str = "low quality, blurry",
        qc_prompts: Optional[Tuple[str, str]] = None,
        num_frames: int = 17,
        n_choices: int = 8,
        min_steps: float = 0.3,
        max_steps: float = 0.55,
        ddim_steps: int = 250,
        guide_scale: float = 7.5,
        optimize_cond: int = 0,
        cond_lr: float = 1e-4,
        latent_interp: str = 'slerp',
        schedule_type: str = 'linear',
        out_dir: str = 'output',
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Setup output directory
        os.makedirs(out_dir, exist_ok=True)
        img1.save(f'{out_dir}/000.png')
        img2.save(f'{out_dir}/{num_frames-1:03d}.png')

        # Validate num_frames
        num_levels = int(np.log2(num_frames - 1))
        assert 2**num_levels == num_frames - 1, \
            f"num_frames-1 must be power of 2, got {num_frames-1}"

        print(f"\n{'='*70}")
        print(f"Starting Hierarchical Interpolation")
        print(f"{'='*70}")
        print(f"Frames: {num_frames} ({num_levels} levels)")
        print(f"DDIM steps: {ddim_steps}")
        print(f"Candidates per frame: {n_choices}")
        print(f"Quality control: {'CLIP' if qc_prompts else 'Manual'}")
        print(f"Textual inversion: {optimize_cond} iters" if optimize_cond else "Textual inversion: Disabled")
        print(f"{'='*70}\n")

        # Convert min/max_steps to absolute if needed
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)

        # Setup DDIM scheduler
        self.scheduler.set_timesteps(ddim_steps)
        timesteps = self.scheduler.timesteps.cpu().numpy()

        # Get step schedule for hierarchy
        step_schedule = get_step_schedule(
            int(min_steps), int(max_steps), num_levels, schedule_type
        )

        print(f"Step schedule: {step_schedule}")
        print(f"Timestep range: {timesteps[step_schedule[-1]]} → {timesteps[step_schedule[1]]}\n")

        # Textual inversion (optional)
        if optimize_cond > 0:
            h = hashlib.md5()
            h.update(np.array(img1).tobytes())
            h.update(np.array(img2).tobytes())
            h.update(f"{prompt}|{n_prompt}|{optimize_cond}|{cond_lr}|{guide_scale}|{self.model_id}".encode())
            cache_path = os.path.join(out_dir, f"embeddings_{h.hexdigest()}.pt")

            cond1, cond2, uncond = self._optimize_embeddings(
                img1, img2, prompt, n_prompt,
                num_iters=optimize_cond, lr=cond_lr, guide_scale=guide_scale,
                cache_path=cache_path,
            )
            print()
        else:
            print("Using base text embeddings (no optimization)...")
            cond1 = self._encode_prompt(prompt)
            cond2 = cond1.clone()
            uncond = self._encode_prompt(n_prompt)
            print()

        # Main inference (no gradients needed)
        with torch.no_grad():
            # Encode endpoint images
            print("Encoding endpoint images...")
            final_latents = [None] * num_frames
            final_latents[0] = self._encode_image(img1)
            final_latents[-1] = self._encode_image(img2)
            print(f"Latent shape: {final_latents[0].shape}\n")

            # Choose interpolation function
            interp_fn = slerp if latent_interp == 'slerp' else interpolate_linear

            # Hierarchical generation
            print(f"{'='*70}")
            print("Hierarchical Generation")
            print(f"{'='*70}\n")

            for level in range(1, num_levels + 1):
                cur_step = step_schedule[-level]
                prev_step = step_schedule[-level - 1] if level < num_levels else 0
                t = int(timesteps[cur_step])
                df = 2 ** (num_levels - level)  # Frame step at this level

                print(f"Level {level}/{num_levels}:")
                print(f"  Timestep: {t}")
                print(f"  Frame step: {df}")
                print(f"  Denoising range: timesteps[{prev_step}:{cur_step}]")
                print(f"  Frames to generate: {len(range(df, num_frames-1, df*2))}")

                for frame_ix in range(df, num_frames - 1, df * 2):
                    frac = frame_ix / (num_frames - 1)

                    print(f"    Generating frame {frame_ix}/{num_frames-1} (α={frac:.3f})...")

                    # Interpolate text conditioning
                    cond_interp = interp_fn(cond1, cond2, frac)
                    text_emb = torch.cat([uncond, cond_interp])

                    # Generate N candidates
                    candidates = []
                    clip_scores = []

                    for choice_ix in range(n_choices):
                        # Add noise to neighboring final_latents
                        noise = torch.randn_like(final_latents[0])

                        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
                        sqrt_alpha = (alphas_cumprod[t] ** 0.5).item()
                        sqrt_one_minus = ((1 - alphas_cumprod[t]) ** 0.5).item()

                        print(f" t {t} -> sqrt_alpha {sqrt_alpha:.4f}, sqrt_one_minus {sqrt_one_minus:.4f}")

                        # Noise the neighbors
                        l1 = sqrt_alpha * final_latents[frame_ix - df] + sqrt_one_minus * noise
                        l2 = sqrt_alpha * final_latents[frame_ix + df] + sqrt_one_minus * noise

                        # Interpolate
                        noisy_latent = interp_fn(l1, l2, 0.5)

                        # Denoise: CRITICAL FIX - denoise from cur_step all the way to timestep 0
                        # This matches cm.py:570-574 where ddim_sampler.sample(timesteps=cur_step)
                        # denoises from cur_step to fully clean (timestep 0)
                        for step_idx in range(cur_step, len(timesteps)):
                            t_cur = int(timesteps[step_idx])
                            print(f"      Denoising step {step_idx}/{len(timesteps)-1} (t={t_cur})...")
                            noisy_latent = self._denoise_step(
                                noisy_latent, text_emb, t_cur, guide_scale
                            )

                        candidates.append(noisy_latent)

                        # Decode and evaluate
                        image = self._decode_latent(noisy_latent)

                        if qc_prompts:
                            # Automatic CLIP scoring
                            score = self._evaluate_with_clip(
                                image, qc_prompts[0], qc_prompts[1]
                            )
                            clip_scores.append(score)
                        else:
                            # Manual selection - save all candidates
                            image.save(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png')

                    # Select best candidate
                    if qc_prompts:
                        best_idx = int(np.argmax(clip_scores))
                        print(f"      Selected candidate {best_idx} "
                            f"(score: {clip_scores[best_idx]:.4f})")

                        image = self._decode_latent(candidates[best_idx])
                        image.save(f'{out_dir}/{frame_ix:03d}.png')

                    else:
                        # Manual selection
                        print(f"      Saved {n_choices} candidates. Choose 0-{n_choices-1}:")
                        choice = int(input("      Choice: "))
                        best_idx = choice

                        # Clean up non-selected candidates
                        for i in range(n_choices):
                            if i != choice:
                                os.remove(f'{out_dir}/{frame_ix:03d}_{i}.png')
                            else:
                                os.rename(
                                    f'{out_dir}/{frame_ix:03d}_{i}.png',
                                    f'{out_dir}/{frame_ix:03d}.png'
                                )

                    # Cache the selected latent
                    final_latents[frame_ix] = candidates[best_idx]

            # Reduce choices at finer levels
            n_choices = max(n_choices - 1, 3)

            print()

        # Decode all final frames
        print(f"{'='*70}")
        print("Decoding Final Sequence")
        print(f"{'='*70}\n")

        result_images = []
        for i in trange(num_frames, desc="Decoding frames"):
            if i == 0:
                result_images.append(img1)
            elif i == num_frames - 1:
                result_images.append(img2)
            else:
                # Check if already saved
                if os.path.exists(f'{out_dir}/{i:03d}.png'):
                    img = Image.open(f'{out_dir}/{i:03d}.png')
                else:
                    img = self._decode_latent(final_latents[i])
                    img.save(f'{out_dir}/{i:03d}.png')
                result_images.append(img)

        print(f"\n✓ Interpolation complete!")
        print(f"✓ {num_frames} frames saved to {out_dir}/")

        return result_images


# ============================================================================
# Helper Functions (copied from cm.py)
# ============================================================================

def get_step_schedule(
    min_steps: int,
    max_steps: int,
    num_levels: int,
    schedule_type: str = 'linear'
) -> List[int]:
    """
    Generate step schedule for hierarchical interpolation.

    Args:
        min_steps: Starting step index
        max_steps: Ending step index
        num_levels: Number of hierarchy levels
        schedule_type: 'linear', 'convex', or 'concave'

    Returns:
        List of step indices [0, step1, step2, ..., stepN]

    Example:
        >>> get_step_schedule(50, 125, 3, 'linear')
        [0, 50, 87, 125]
    """
    diff = max_steps - min_steps

    if schedule_type == 'concave':
        return [0] + [
            int(diff * x**0.5) + min_steps
            for x in np.linspace(0, 1, num_levels)
        ]
    elif schedule_type == 'convex':
        return [0] + [
            int(diff * x**2) + min_steps
            for x in np.linspace(0, 1, num_levels)
        ]
    elif schedule_type == 'linear':
        return [0] + [
            int(x) for x in np.linspace(min_steps, max_steps, num_levels)
        ]
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


@torch.no_grad()
def slerp(p0: torch.Tensor, p1: torch.Tensor, fract_mixing: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Preserves the norm of the interpolated vector, which is important
    for interpolating in latent spaces.

    Args:
        p0: First tensor
        p1: Second tensor
        fract_mixing: Mixing coefficient in [0, 1]
                     0 returns p0, 1 returns p1

    Returns:
        Interpolated tensor
    """
    # Determine dtype for recasting
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    # Cast to highest precision available
    # MPS doesn't support float64, so use float32
    if p0.device.type == 'mps':
        # MPS: stay in float32
        p0 = p0.float()
        p1 = p1.float()
    else:
        # CUDA/CPU: use float64 for better numerical stability
        p0 = p0.double()
        p1 = p1.double()

    # Compute norms
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7

    # Compute angle between vectors
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Interpolate
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    # Recast to original dtype
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    # If we used double precision, convert back to float32
    elif interp.dtype == torch.float64:
        interp = interp.float()

    return interp


def interpolate_linear(p0: torch.Tensor, p1: torch.Tensor, frac: float) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    Args:
        p0: First tensor
        p1: Second tensor
        frac: Mixing coefficient in [0, 1]

    Returns:
        Interpolated tensor
    """
    return p0 + (p1 - p0) * frac


# ============================================================================
# Main script for testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing DiffusersInterpolator - Phase 1: Setup & Model Loading")
    print("=" * 70)

    # Check available devices
    print("\nAvailable devices:")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  MPS: {torch.backends.mps.is_available()}")
    print(f"  CPU: Always available")

    # Initialize (auto-detects device)
    interpolator = DiffusersInterpolator(
        model_id="runwayml/stable-diffusion-v1-5"
    )

    print("\n" + "=" * 70)
    print("Testing encode/decode")
    print("=" * 70)

    # Create a test image
    test_img = Image.new('RGB', (512, 512), color='red')

    # Encode
    print("Encoding test image...")
    latent = interpolator._encode_image(test_img)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent dtype: {latent.dtype}")
    print(f"  Latent range: [{latent.min():.4f}, {latent.max():.4f}]")

    # Decode
    print("\nDecoding latent...")
    reconstructed = interpolator._decode_latent(latent)
    print(f"  Reconstructed size: {reconstructed.size}")

    reconstructed.save('/tmp/test_reconstruction.png')
    print(f"  Saved to: /tmp/test_reconstruction.png")

    # Test text encoding
    print("\n" + "=" * 70)
    print("Testing text encoding")
    print("=" * 70)

    prompt = "a photograph of a red apple"
    text_emb = interpolator._encode_prompt(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Embedding shape: {text_emb.shape}")
    print(f"  Embedding dtype: {text_emb.dtype}")

    # Test noise scheduling
    print("\n" + "=" * 70)
    print("Testing noise scheduling")
    print("=" * 70)

    timestep_schedule = [0, 250, 500, 750]
    latents1, latents2 = interpolator._get_latent_stack(
        test_img, test_img, timestep_schedule, share_noise=True
    )
    print(f"  Timesteps: {timestep_schedule}")
    print(f"  Latent stack length: {len(latents1)}")
    print(f"  Each latent shape: {latents1[0].shape}")

    print("\n✓ All phases complete!")
    print("\n" + "=" * 70)
    print("Ready for interpolation!")
    print("=" * 70)
    print("\nTo run a full interpolation, see: examples/basic_interpolation.py")

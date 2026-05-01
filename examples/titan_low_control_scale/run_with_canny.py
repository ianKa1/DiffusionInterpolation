"""
Armored Titan to Attack Titan interpolation with Canny ControlNet.

Interpolates between armored_titan.png and attack_titan.png using Canny edge
detection to maintain structural consistency across 33 frames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from diffusers_interpolate_qc import DiffusersInterpolator

print("Initializing interpolator with Canny ControlNet...")
interpolator = DiffusersInterpolator(
    model_id="runwayml/stable-diffusion-v1-5",
    controlnet_model="lllyasviel/sd-controlnet-canny"
)

print("\nLoading images...")
script_dir = os.path.dirname(os.path.abspath(__file__))
img1 = Image.open(os.path.join(script_dir, 'armored_titan.png')).convert('RGB').resize((512, 512))
img2 = Image.open(os.path.join(script_dir, 'attack_titan.png')).convert('RGB').resize((512, 512))

print(f"Image 1 (armored titan): {img1.size}")
print(f"Image 2 (attack titan): {img2.size}")

prompt = (
    'armored titan, heavy armor plating, powerful, anime, detailed, high quality, sharp',
    'attack titan, muscular, fierce, anime, detailed, high quality, sharp'
)
n_prompt = (
    'blurry, low quality, distorted, ugly, watermark, text',
    'blurry, low quality, distorted, ugly, watermark, text'
)

qc_prompt = 'titan, anime, detailed, high quality, clear, sharp'
qc_neg_prompt = 'blurry, distorted, low quality, ugly, artifacts, watermark'

output_dir = os.path.join(script_dir, 'output')

# 33 frames: 32 intervals = 2^5 levels
print("\nRunning interpolation with Canny ControlNet...")
print("33 frames (32 intervals = 2^5 levels)")

results = interpolator.interpolate_qc(
    img1, img2,
    prompt=prompt,
    n_prompt=n_prompt,
    qc_prompts=(qc_prompt, qc_neg_prompt),
    num_frames=33,
    n_choices=4,
    ddim_steps=200,
    min_steps=0.3,
    max_steps=0.55,
    optimize_cond=0,
    use_controlnet=True,
    controlnet_conditioning_scale=0.6,
    latent_interp='slerp',
    schedule_type='linear',
    out_dir=output_dir,
    seed=42
)

print(f"\n✓ Done! Generated {len(results)} frames in {output_dir}/")

print("\nCreating montage (showing every 2nd frame)...")
sample_indices = list(range(0, len(results), 2))
thumb_size = 192
n_cols = 8
n_rows = (len(sample_indices) + n_cols - 1) // n_cols

montage = Image.new('RGB', (thumb_size * n_cols, thumb_size * n_rows), (255, 255, 255))

for idx, frame_num in enumerate(sample_indices):
    row = idx // n_cols
    col = idx % n_cols
    thumb = results[frame_num].resize((thumb_size, thumb_size))
    montage.paste(thumb, (col * thumb_size, row * thumb_size))

montage_path = os.path.join(output_dir, 'montage.png')
montage.save(montage_path)
print(f"✓ Saved montage to {montage_path}")

print("\nCreating animated GIF...")
gif_path = os.path.join(output_dir, 'armored_to_attack_titan.gif')
results[0].save(
    gif_path,
    save_all=True,
    append_images=results[1:],
    duration=100,
    loop=0
)
print(f"✓ Saved animated GIF to {gif_path}")

print(f"\n{'='*70}")
print("Armored Titan → Attack Titan Interpolation Complete!")
print(f"{'='*70}")
print(f"Total frames: {len(results)}")
print(f"Output directory: {output_dir}/")
print(f"Montage: {montage_path}")
print(f"Animated GIF: {gif_path}")
print(f"{'='*70}")

"""
Two-man pose interpolation: Attack on Titan to Matrix with OpenPose ControlNet.

Interpolates between attack_on_titan.jpg and matrix.jpg using OpenPose
to maintain pose consistency across 33 frames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from diffusers_interpolate_qc import DiffusersInterpolator


def crop_to_square(image: Image.Image) -> Image.Image:
    """Crop image to square by removing left and right sides (center crop)."""
    width, height = image.size

    if width > height:
        # Crop left and right sides
        left = (width - height) // 2
        right = left + height
        return image.crop((left, 0, right, height))
    elif height > width:
        # Crop top and bottom sides
        top = (height - width) // 2
        bottom = top + width
        return image.crop((0, top, width, bottom))
    else:
        # Already square
        return image


# Initialize with OpenPose ControlNet
print("Initializing interpolator with OpenPose ControlNet...")
interpolator = DiffusersInterpolator(
    model_id="runwayml/stable-diffusion-v1-5",
    controlnet_model="lllyasviel/control_v11p_sd15_openpose"
)

# Load and prepare images
print("\nLoading images...")
script_dir = os.path.dirname(os.path.abspath(__file__))

img1_raw = Image.open(os.path.join(script_dir, 'attack_on_titan.jpg')).convert('RGB')
img2_raw = Image.open(os.path.join(script_dir, 'matrix.jpg')).convert('RGB')

print(f"Original image 1 size: {img1_raw.size}")
print(f"Original image 2 size: {img2_raw.size}")

# Crop to square and resize
img1 = crop_to_square(img1_raw).resize((512, 512))
img2 = crop_to_square(img2_raw).resize((512, 512))

print(f"Processed image 1 size: {img1.size}")
print(f"Processed image 2 size: {img2.size}")

# Prompts - different for each image
prompt1 = 'anime character, soldier, military uniform, action pose, attack on titan style, detailed, high quality, illustration'
prompt2 = 'person in black coat, sunglasses, matrix style, action pose, cinematic, photorealistic, detailed, high quality'

n_prompt1 = 'blurry, low quality, distorted, ugly, watermark, text, multiple people, photo, realistic'
n_prompt2 = 'blurry, low quality, distorted, ugly, watermark, text, multiple people, cartoon, anime, illustration'

qc_prompt = 'person portrait, detailed, high quality, clear, action pose'
qc_neg_prompt = 'blurry, distorted, low quality, ugly, artifacts, multiple people'

# Run interpolation with 33 frames (32 intervals = 2^5)
print("\nRunning interpolation with OpenPose ControlNet...")
print("33 frames (32 intervals = 2^5 levels)")

output_dir = os.path.join(script_dir, 'output')

results = interpolator.interpolate_qc(
    img1, img2,
    prompt=(prompt1, prompt2),  # Different prompts: anime style -> realistic style
    n_prompt=(n_prompt1, n_prompt2),  # Different negative prompts for each style
    qc_prompts=(qc_prompt, qc_neg_prompt),
    num_frames=33,          # 33 frames (32 intervals = 2^5)
    n_choices=4,            # 4 candidates per frame
    ddim_steps=200,         # 200 steps for high quality
    min_steps=0.3,          # 30% noise minimum
    max_steps=0.55,         # 55% noise maximum
    optimize_cond=0,      # Textual inversion for quality
    use_controlnet=True,    # Enable ControlNet
    controlnet_conditioning_scale=1.5,  # Dynamic strength (1.0 at endpoints, 1.5 at midpoint)
    latent_interp='slerp',  # Spherical interpolation for latents
    schedule_type='linear', # Linear noise schedule
    out_dir=output_dir,
    seed=42
)

print(f"\n✓ Done! Generated {len(results)} frames in {output_dir}/")

# Create montage (every 2nd frame to keep it manageable)
print("\nCreating montage (showing every 2nd frame)...")
from PIL import ImageDraw, ImageFont

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

# Create animated GIF
print("\nCreating animated GIF...")
gif_path = os.path.join(output_dir, 'attack_to_matrix.gif')
results[0].save(
    gif_path,
    save_all=True,
    append_images=results[1:],
    duration=100,  # milliseconds per frame (10 FPS)
    loop=0  # infinite loop
)
print(f"✓ Saved animated GIF to {gif_path}")

print(f"\n{'='*70}")
print("Attack on Titan → Matrix Interpolation Complete!")
print(f"{'='*70}")
print(f"Total frames: {len(results)}")
print(f"Output directory: {output_dir}/")
print(f"Montage: {montage_path}")
print(f"Animated GIF: {gif_path}")
print(f"{'='*70}")

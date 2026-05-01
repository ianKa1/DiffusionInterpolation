"""
Waterfall interpolation: Longhair to Waterfall.

Interpolates between longhair.png and waterfall.png without ControlNet
across 33 frames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from diffusers_interpolate_qc import DiffusersInterpolator

# Initialize without ControlNet
print("Initializing interpolator...")
interpolator = DiffusersInterpolator(
    model_id="runwayml/stable-diffusion-v1-5"
)

# Load images
print("\nLoading images...")
script_dir = os.path.dirname(os.path.abspath(__file__))
img1 = Image.open(os.path.join(script_dir, 'longhair.png')).convert('RGB').resize((512, 512))
img2 = Image.open(os.path.join(script_dir, 'waterfall.png')).convert('RGB').resize((512, 512))

print(f"Image 1 (longhair): {img1.size}")
print(f"Image 2 (waterfall): {img2.size}")

# Prompts - different for each image
prompt1 = 'portrait, woman with long hair, beautiful face, close-up, high quality, detailed, photorealistic'
prompt2 = 'waterfall, flowing water, nature scene, landscape, beautiful scenery, high quality, detailed, photorealistic'

n_prompt1 = 'blurry, low quality, distorted, ugly, watermark, text, bad hair'
n_prompt2 = 'blurry, low quality, distorted, ugly, watermark, text, dry, desert'

qc_prompt = 'beautiful, detailed, high quality, clear, photorealistic'
qc_neg_prompt = 'blurry, distorted, low quality, ugly, artifacts'

# Run interpolation with 33 frames (32 intervals = 2^5)
print("\nRunning interpolation...")
print("33 frames (32 intervals = 2^5 levels)")

output_dir = os.path.join(script_dir, 'output')

results = interpolator.interpolate_qc(
    img1, img2,
    prompt=(prompt1, prompt2),  # Different prompts for each image
    n_prompt=(n_prompt1, n_prompt2),  # Different negative prompts for each image
    qc_prompts=(qc_prompt, qc_neg_prompt),
    num_frames=33,          # 33 frames (32 intervals = 2^5)
    n_choices=4,            # 4 candidates per frame
    ddim_steps=200,         # 200 steps for high quality
    min_steps=0.3,          # 30% noise minimum
    max_steps=0.55,         # 55% noise maximum
    optimize_cond=0,      # Textual inversion for quality
    use_controlnet=False,   # No ControlNet
    latent_interp='slerp',  # Spherical interpolation for latents
    schedule_type='linear', # Linear noise schedule
    out_dir=output_dir,
    seed=42
)

print(f"\n✓ Done! Generated {len(results)} frames in {output_dir}/")

# Create montage (every 2nd frame to keep it manageable)
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

# Create animated GIF
print("\nCreating animated GIF...")
gif_path = os.path.join(output_dir, 'longhair_to_waterfall.gif')
results[0].save(
    gif_path,
    save_all=True,
    append_images=results[1:],
    duration=100,  # milliseconds per frame (10 FPS)
    loop=0  # infinite loop
)
print(f"✓ Saved animated GIF to {gif_path}")

print(f"\n{'='*70}")
print("Longhair → Waterfall Interpolation Complete!")
print(f"{'='*70}")
print(f"Total frames: {len(results)}")
print(f"Output directory: {output_dir}/")
print(f"Montage: {montage_path}")
print(f"Animated GIF: {gif_path}")
print(f"{'='*70}")

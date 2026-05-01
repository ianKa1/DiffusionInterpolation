"""
Scenery interpolation: Spring to Winter with Canny ControlNet.

Interpolates between spring.png and winter.png using Canny edge detection
to maintain structural consistency across 64 frames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from diffusers_interpolate_qc import DiffusersInterpolator

# Initialize with Canny ControlNet
print("Initializing interpolator with Canny ControlNet...")
interpolator = DiffusersInterpolator(
    model_id="runwayml/stable-diffusion-v1-5",
    controlnet_model="lllyasviel/sd-controlnet-canny"
)

# Load spring and winter images
print("\nLoading images...")
script_dir = os.path.dirname(os.path.abspath(__file__))
img1 = Image.open(os.path.join(script_dir, 'spring.png')).convert('RGB').resize((512, 512))
img2 = Image.open(os.path.join(script_dir, 'winter.png')).convert('RGB').resize((512, 512))

print(f"Image 1 (spring): {img1.size}")
print(f"Image 2 (winter): {img2.size}")

# Prompts for scenery
prompt = 'landscape, scenery, nature, high quality, detailed, photorealistic, 4k'
n_prompt = 'blurry, low quality, distorted, ugly, watermark, text'

qc_prompt = 'beautiful landscape, detailed scenery, high quality, nature'
qc_neg_prompt = 'blurry, distorted, low quality, ugly, artifacts'

# Run interpolation with 64 frames (63 intervals = no power of 2!)
# Need 65 frames (64 intervals = 2^6)
print("\nRunning interpolation with Canny ControlNet...")
print("65 frames (64 intervals = 2^6 levels)")

results = interpolator.interpolate_qc(
    img1, img2,
    prompt=prompt,
    n_prompt=n_prompt,
    qc_prompts=(qc_prompt, qc_neg_prompt),  # Automatic CLIP selection
    num_frames=65,          # 65 frames (64 intervals = 2^6)
    n_choices=4,            # 5 candidates per frame
    ddim_steps=200,          # 200 steps (use 200+ for higher quality)
    min_steps=0.3,          # 30% noise minimum
    max_steps=0.55,         # 55% noise maximum
    optimize_cond=0,        # No textual inversion (use 200+ for quality)
    use_controlnet=True,    # Enable ControlNet
    controlnet_conditioning_scale=1.5,  # Dynamic strength (1.0 at endpoints, 1.5 at midpoint)
    latent_interp='slerp',  # Spherical interpolation for latents
    schedule_type='linear', # Linear noise schedule
    out_dir='./output',
    seed=42
)

print(f"\n✓ Done! Generated {len(results)} frames in ./output/")

# Create montage (every 4th frame to keep it manageable)
print("\nCreating montage (showing every 4th frame)...")
from PIL import ImageDraw, ImageFont

sample_indices = list(range(0, len(results), 4))
thumb_size = 192
n_cols = 8
n_rows = (len(sample_indices) + n_cols - 1) // n_cols

montage = Image.new('RGB', (thumb_size * n_cols, thumb_size * n_rows), (255, 255, 255))

for idx, frame_num in enumerate(sample_indices):
    row = idx // n_cols
    col = idx % n_cols

    thumb = results[frame_num].resize((thumb_size, thumb_size))
    montage.paste(thumb, (col * thumb_size, row * thumb_size))

montage.save('./output/montage.png')
print("✓ Saved montage to ./output/montage.png")

# Create animated GIF
print("\nCreating animated GIF...")
results[0].save(
    './output/spring_to_winter.gif',
    save_all=True,
    append_images=results[1:],
    duration=100,  # milliseconds per frame (10 FPS)
    loop=0  # infinite loop
)
print("✓ Saved animated GIF to ./output/spring_to_winter.gif")

print(f"\n{'='*70}")
print("Spring → Winter Interpolation Complete!")
print(f"{'='*70}")
print(f"Total frames: {len(results)}")
print(f"Output directory: ./output/")
print(f"Montage: ./output/montage.png")
print(f"Animated GIF: ./output/spring_to_winter.gif")
print(f"{'='*70}")

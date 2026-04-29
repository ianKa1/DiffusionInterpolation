"""
Advanced example: Interpolation with textual inversion (optimized embeddings)

This produces higher quality results by optimizing text embeddings for each image.
Slower but much better quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from diffusers_interpolate_qc import DiffusersInterpolator

# Initialize
print("Initializing interpolator...")
interpolator = DiffusersInterpolator(
    model_id="runwayml/stable-diffusion-v1-5"
)

# Load images
print("\nLoading images...")
# Replace these with your own images
img1 = Image.open('../sample_imgs/noface1.png').convert('RGB').resize((512, 512))
img2 = Image.open('../sample_imgs/noface2.png').convert('RGB').resize((512, 512))

# Run interpolation with textual inversion
print("\nRunning interpolation with textual inversion...")
print("Note: This will take ~10-15 minutes with optimization")

prompt = 'portrait, cartoon, mask, ghost, high resolution, highly detailed, ultra HD, 4k, simple, elegant'
n_prompt = 'lowres, messy, lopsided, disfigured, low quality, photo'

qc_prompt = 'portrait, cartoon, mask, detailed, high quality, simple, elegant'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, photo, low quality, multiple faces'

results = interpolator.interpolate_qc(
    img1, img2,
    prompt=prompt,
    n_prompt=n_prompt,
    qc_prompts=(qc_prompt, qc_neg_prompt),
    num_frames=17,          # 17 frames (16 intervals = 2^4)
    n_choices=5,            # Generate 5 candidates per frame
    ddim_steps=200,         # High quality
    min_steps=0.25,         # 25% noise minimum
    max_steps=0.6,          # 60% noise maximum
    optimize_cond=200,      # 200 iterations of textual inversion
    cond_lr=1e-4,           # Learning rate
    latent_interp='slerp',
    out_dir='output_textual_inversion',
    seed=42
)

print(f"\n✓ Done! Generated {len(results)} frames in output_textual_inversion/")

# Create montage
print("\nCreating montage...")
montage_width = 512 * min(len(results), 8)
montage_height = 512 * ((len(results) - 1) // 8 + 1)
montage = Image.new('RGB', (montage_width, montage_height))

for i, img in enumerate(results):
    row = i // 8
    col = i % 8
    montage.paste(img, (col * 512, row * 512))

montage.save('output_textual_inversion/montage.png')
print("✓ Saved montage to output_textual_inversion/montage.png")

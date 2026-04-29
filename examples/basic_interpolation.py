"""
Basic example: Interpolate between two images using diffusers_interpolate_qc.py

This is the simplest usage - no textual inversion, no CLIP QC, just basic hierarchical interpolation.
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

# Load images (replace with your own images)
print("\nLoading images...")
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img1 = Image.open(os.path.join(_repo, 'sample_imgs', 'noface1.png')).convert('RGB').resize((512, 512))
img2 = Image.open(os.path.join(_repo, 'sample_imgs', 'noface2.jpeg')).convert('RGB').resize((512, 512))

prompt = 'portrait, cartoon, mask, ghost, high resolution, highly detailed, ultra HD, 4k, simple, elegant'
n_prompt = 'lowres, messy, lopsided, disfigured, low quality, photo'

qc_prompt = 'portrait, cartoon, mask, detailed, high quality, simple, elegant'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, photo, low quality, multiple faces'

# Run interpolation
print("\nRunning interpolation...")
results = interpolator.interpolate_qc(
    img1, img2,
    prompt=prompt,
    n_prompt=n_prompt,
    qc_prompts=(qc_prompt, qc_neg_prompt),  # Automatic CLIP selection
    num_frames=9,           # 9 frames (8 intervals = 2^3)
    n_choices=3,            # Generate 3 candidates per frame
    ddim_steps=50,          # Faster (use 200 for quality)
    min_steps=0.3,          # 30% noise minimum
    max_steps=0.5,          # 50% noise maximum
    optimize_cond=0,        # Skip textual inversion for speed (use 200 for quality)
    latent_interp='slerp',  # Spherical interpolation
    out_dir='output_basic',
    seed=42                 # Reproducibility
)

print(f"\n✓ Done! Generated {len(results)} frames in output_basic/")

# Create a contact sheet
from PIL import ImageDraw, ImageFont

print("\nCreating contact sheet...")
thumb_size = 128
contact_sheet = Image.new('RGB', (thumb_size * len(results), thumb_size))

for i, img in enumerate(results):
    thumb = img.resize((thumb_size, thumb_size))
    contact_sheet.paste(thumb, (i * thumb_size, 0))

contact_sheet.save('output_basic/contact_sheet.png')
print("✓ Saved contact sheet to output_basic/contact_sheet.png")

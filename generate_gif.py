#!/usr/bin/env python3
"""
Generate GIF from interpolation output frames.

Usage:
    python generate_gif.py <directory_name> [options]
    python generate_gif.py dog_sketch --fps 10 --pause 1.0

Author: Helper script for DiffusionInterpolation
"""

import argparse
import os
import glob
from pathlib import Path
from PIL import Image


def generate_gif(
    example_dir: str,
    output_name: str = None,
    fps: int = 10,
    pause_duration: float = 0.5,
    loop: int = 0,
    optimize: bool = True,
    quality: int = 95
):
    """
    Generate GIF from PNG sequence in examples/<example_dir>/output/

    Args:
        example_dir: Name of directory in examples/
        output_name: Name of output GIF (default: <example_dir>.gif)
        fps: Frames per second (default: 10)
        pause_duration: Pause after last frame in seconds (default: 0.5)
        loop: Number of loops (0 = infinite, default: 0)
        optimize: Optimize GIF file size (default: True)
        quality: Quality for optimization 1-100 (default: 95)
    """
    # Construct paths
    base_path = Path(__file__).parent
    output_dir = base_path / "examples" / example_dir / "output"

    if not output_dir.exists():
        raise ValueError(f"Directory does not exist: {output_dir}")

    # Find all numbered PNG files (000.png, 001.png, etc.)
    # Exclude control images and other non-frame files
    png_files = sorted(glob.glob(str(output_dir / "[0-9][0-9][0-9].png")))

    if not png_files:
        raise ValueError(f"No numbered PNG files found in {output_dir}")

    print(f"Found {len(png_files)} frames in {output_dir}")
    print(f"Frame range: {Path(png_files[0]).name} to {Path(png_files[-1]).name}")

    # Load images
    frames = []
    for png_file in png_files:
        img = Image.open(png_file)
        # Convert to RGB if necessary (GIF doesn't support RGBA well)
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        frames.append(img)

    print(f"Loaded {len(frames)} frames")
    print(f"Frame size: {frames[0].size}")

    # Calculate frame duration in milliseconds
    frame_duration = int(1000 / fps)

    # Add pause at the end by duplicating the last frame
    if pause_duration > 0:
        num_pause_frames = int(pause_duration * fps)
        print(f"Adding {num_pause_frames} pause frames at the end ({pause_duration}s)")
        last_frame = frames[-1]
        for _ in range(num_pause_frames):
            frames.append(last_frame)

    # Determine output path
    if output_name is None:
        output_name = f"{example_dir}.gif"

    if not output_name.endswith('.gif'):
        output_name += '.gif'

    output_path = output_dir / output_name

    # Save GIF
    print(f"\nGenerating GIF...")
    print(f"  FPS: {fps}")
    print(f"  Frame duration: {frame_duration}ms")
    print(f"  Total frames (with pause): {len(frames)}")
    print(f"  Loop: {'infinite' if loop == 0 else loop}")
    print(f"  Optimize: {optimize}")
    if optimize:
        print(f"  Quality: {quality}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop,
        optimize=optimize,
        quality=quality
    )

    # Get file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"\n✓ GIF saved to: {output_path}")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"✓ Duration: {len(frames) / fps:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Generate GIF from interpolation output frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GIF with default settings (10 fps, 0.5s pause)
  python generate_gif.py dog_sketch

  # Custom frame rate and pause
  python generate_gif.py scenery --fps 15 --pause 1.0

  # High quality, no optimization
  python generate_gif.py waterfall --fps 20 --no-optimize --quality 100

  # Custom output name
  python generate_gif.py two_man2 --output animation.gif

  # Process all example directories
  for dir in examples/*/; do
      python generate_gif.py $(basename $dir)
  done
        """
    )

    parser.add_argument(
        'directory',
        help='Directory name in examples/ (e.g., dog_sketch, scenery)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output GIF filename (default: <directory>.gif)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )

    parser.add_argument(
        '--pause',
        type=float,
        default=0.5,
        help='Pause duration after last frame in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--loop',
        type=int,
        default=0,
        help='Number of loops (0 = infinite, default: 0)'
    )

    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable GIF optimization (larger file size, faster generation)'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='Quality for optimization 1-100 (default: 95, only used if optimizing)'
    )

    args = parser.parse_args()

    try:
        generate_gif(
            example_dir=args.directory,
            output_name=args.output,
            fps=args.fps,
            pause_duration=args.pause,
            loop=args.loop,
            optimize=not args.no_optimize,
            quality=args.quality
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

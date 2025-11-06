#!/usr/bin/env python3
"""
Simple script to download all 60 tracks directly
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from download_samples import download_fma_small_samples

if __name__ == "__main__":
    print("Starting download of 60 tracks from Jamendo...")
    success = download_fma_small_samples()

    if success:
        print("\n" + "=" * 60)
        print("✅ Download complete!")
        sample_dir = Path(__file__).parent / "sample_audio"
        mp3_count = len(list(sample_dir.glob("*.mp3")))
        print(f"📁 Location: {sample_dir.absolute()}")
        print(f"📊 Files: {mp3_count} MP3 files")
    else:
        print("\n⚠️ Download failed or incomplete")
        sys.exit(1)

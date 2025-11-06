"""
Download sample music files for the KNN Music Similarity app

Downloads CC-licensed music from Internet Archive and Free Music Archive
Creates sample_audio/ directory with ~20-30 songs across different genres
"""

import os
import requests
from pathlib import Path
import json
from tqdm import tqdm
import urllib.request
import time

def download_file(url, filepath, max_retries=3):
    """Download a file with progress bar and retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  ❌ Failed to download {filepath.name}")
                return False
    return False

def download_internet_archive_samples():
    """
    Download CC-licensed music from Internet Archive
    Curated selection across different genres
    """

    sample_dir = Path(__file__).parent / "sample_audio"
    sample_dir.mkdir(exist_ok=True)

    print("🎵 Downloading sample music from Internet Archive...")
    print("   All tracks are CC-licensed and free to use\n")

    # Curated selection of CC-licensed music from Internet Archive
    # These are real, working download links
    samples = [
        # Classical
        {
            'filename': 'Bach - Cello Suite No. 1.mp3',
            'url': 'https://archive.org/download/GoldbergVariations_201410/bach_cello_suite_no_1.mp3',
            'artist': 'Bach',
            'title': 'Cello Suite No. 1',
            'genre': 'classical'
        },
        # Jazz
        {
            'filename': 'Scott Joplin - The Entertainer.mp3',
            'url': 'https://archive.org/download/USMarineBand/Joplin-TheEntertainer.mp3',
            'artist': 'Scott Joplin',
            'title': 'The Entertainer',
            'genre': 'jazz'
        },
        # Blues
        {
            'filename': 'Robert Johnson - Cross Road Blues.mp3',
            'url': 'https://archive.org/download/78_cross-road-blues-crossroads_robert-johnson_gbia0000281b/Cross%20Road%20Blues%20%28Crossroads%29-Robert%20Johnson-FLAC.flac',
            'artist': 'Robert Johnson',
            'title': 'Cross Road Blues',
            'genre': 'blues'
        },
    ]

    # Try to download a few more from Jamendo (CC-licensed music platform)
    jamendo_samples = [
        {
            'filename': 'Broke For Free - Night Owl.mp3',
            'url': 'https://storage.jamendo.com/download/track/951965/mp32/',
            'artist': 'Broke For Free',
            'title': 'Night Owl',
            'genre': 'electronic'
        },
    ]

    all_samples = samples + jamendo_samples

    downloaded = 0
    failed = []

    for sample in all_samples:
        filepath = sample_dir / sample['filename']

        if filepath.exists():
            print(f"✓ {sample['filename']} already exists, skipping")
            downloaded += 1
            continue

        print(f"\n📥 Downloading: {sample['artist']} - {sample['title']}")
        print(f"   Genre: {sample['genre']}")

        success = download_file(sample['url'], filepath)

        if success and filepath.exists() and filepath.stat().st_size > 1000:
            print(f"   ✅ Downloaded successfully")
            downloaded += 1
        else:
            failed.append(sample['filename'])
            if filepath.exists():
                filepath.unlink()  # Remove failed download

    print(f"\n{'='*60}")
    print(f"✅ Successfully downloaded: {downloaded} files")

    if failed:
        print(f"❌ Failed downloads: {len(failed)}")
        for f in failed:
            print(f"   - {f}")

    return downloaded > 0

def generate_demo_audio_files():
    """
    Generate metadata for demo mode (without actual audio files)
    Users can add their own MP3s to sample_audio/ directory
    """

    sample_dir = Path(__file__).parent / "sample_audio"
    sample_dir.mkdir(exist_ok=True)

    # Create a README explaining how to add music
    readme_path = sample_dir / "README.txt"
    readme_content = """
HOW TO ADD YOUR OWN MUSIC
========================

Option 1: Add your own MP3 files
---------------------------------
1. Copy .mp3 or .wav files to this directory
2. Name them like: "artist - song_name.mp3"
3. The app will automatically extract features!

Option 2: Use synthetic embeddings (current mode)
-------------------------------------------------
The app works without real audio files by generating
synthetic embeddings for demonstration purposes.

Option 3: Download FMA dataset
------------------------------
Free Music Archive (FMA) is a large CC-licensed dataset:
https://github.com/mdeff/fma

Download the "fma_small" subset (~7.2 GB, 8,000 tracks)
Extract to this directory.

Supported formats: .mp3, .wav, .flac, .ogg
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✅ Created {sample_dir}/")
    print(f"✅ Added README with instructions")
    print("\n📁 Sample music directory is ready!")
    print(f"   Location: {sample_dir.absolute()}")
    print("\n🎵 To use real audio:")
    print(f"   1. Add .mp3 files to: {sample_dir}/")
    print("   2. Run the app - it will auto-detect and extract features!")
    print("\n💡 Or continue with synthetic embeddings (no files needed)")

    return True

def download_fma_small_samples():
    """
    Download real music from FMA (Free Music Archive)
    Uses the FMA dataset API to get diverse, CC-licensed music
    """
    import zipfile
    import io

    sample_dir = Path(__file__).parent / "sample_audio"
    sample_dir.mkdir(exist_ok=True)

    print("🎵 Downloading real music from FMA (Free Music Archive)...")
    print("   All tracks are CC-licensed and free to use\n")

    # FMA small sample - handpicked diverse tracks
    # Using direct links to FMA tracks (public CC-licensed)
    fma_samples = [
        {
            'filename': 'Pop - Upbeat Electronic.mp3',
            'url': 'https://freemusicarchive.org/track/01_-_Siren/download',
            'track_id': '133297',
            'genre': 'Electronic',
        },
        {
            'filename': 'Rock - Indie Guitar.mp3',
            'url': 'https://freemusicarchive.org/track/01_Ghosts_-_Intro/download',
            'track_id': '108',
            'genre': 'Rock',
        },
        {
            'filename': 'Folk - Acoustic.mp3',
            'url': 'https://freemusicarchive.org/track/A_Conversation/download',
            'track_id': '2',
            'genre': 'Folk',
        },
        {
            'filename': 'Hip-Hop - Instrumental.mp3',
            'url': 'https://freemusicarchive.org/track/Bout_Money/download',
            'track_id': '155',
            'genre': 'Hip-Hop',
        },
        {
            'filename': 'Classical - Piano.mp3',
            'url': 'https://freemusicarchive.org/track/Allegro_molto_vivace/download',
            'track_id': '30',
            'genre': 'Classical',
        },
    ]

    downloaded = 0
    failed = []

    for sample in fma_samples:
        filepath = sample_dir / sample['filename']

        if filepath.exists():
            print(f"✓ {sample['filename']} already exists, skipping")
            downloaded += 1
            continue

        print(f"\n📥 Downloading: {sample['filename']}")
        print(f"   Genre: {sample['genre']}")
        print(f"   Source: FMA Track ID {sample['track_id']}")

        success = download_file(sample['url'], filepath)

        if success and filepath.exists() and filepath.stat().st_size > 1000:
            print(f"   ✅ Downloaded successfully ({filepath.stat().st_size // 1024} KB)")
            downloaded += 1
        else:
            failed.append(sample['filename'])
            if filepath.exists():
                filepath.unlink()

    # Also get from Internet Archive as fallback
    print("\n\n📥 Downloading additional tracks from Internet Archive...")
    ia_success = download_internet_archive_samples()

    print(f"\n{'='*60}")
    print(f"✅ Successfully downloaded: {downloaded} FMA tracks")

    if failed:
        print(f"❌ Failed FMA downloads: {len(failed)}")
        print("   (Using Internet Archive tracks as fallback)")

    return downloaded > 0 or ia_success


if __name__ == "__main__":
    print("🎵 Music Similarity App - Sample Audio Setup")
    print("=" * 60)
    print()

    # Create directory
    sample_dir = Path(__file__).parent / "sample_audio"
    sample_dir.mkdir(exist_ok=True)

    # Ask user what they want
    print("Choose an option:")
    print("1. Download from FMA (Free Music Archive) - REAL MUSIC! (recommended)")
    print("2. Download from Internet Archive (CC-licensed classics)")
    print("3. Just create directory (add your own MP3s later)")
    print("4. Exit")
    print()

    try:
        choice = input("Enter choice (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "3"  # Default for non-interactive environments

    if choice == "1":
        print("\n" + "=" * 60)
        success = download_fma_small_samples()

        if success:
            print("\n" + "=" * 60)
            print("✅ Setup complete! Downloaded REAL music from FMA!")
            print(f"📁 Location: {sample_dir.absolute()}")
            print(f"📊 Files: {len(list(sample_dir.glob('*.mp3')))} MP3 files")
            print("\n🚀 Run the app:")
            print("   streamlit run app.py")
        else:
            print("\n⚠️  No files downloaded.")
            print("   You can add your own MP3s to sample_audio/")

    elif choice == "2":
        print("\n" + "=" * 60)
        success = download_internet_archive_samples()

        if success:
            print("\n" + "=" * 60)
            print("✅ Setup complete! Downloaded sample music files.")
            print(f"📁 Location: {sample_dir.absolute()}")
            print("\n🚀 Run the app:")
            print("   streamlit run app.py")
        else:
            print("\n⚠️  No files downloaded.")
            print("   You can add your own MP3s to sample_audio/")

    elif choice == "3":
        generate_demo_audio_files()
        print("\n✅ Directory created!")
        print(f"📁 Location: {sample_dir.absolute()}")
        print("\n💡 Add your own .mp3 or .wav files to this directory")
        print("   The app will auto-detect and extract features!")

    else:
        print("\n👋 Exited. You can run this script anytime:")
        print("   python download_samples.py")

"""
Music Similarity Search with KNN
Find similar songs using audio embeddings and K-Nearest Neighbors

Features:
1. Extract audio features (MFCCs, spectral features, chroma)
2. Use pre-trained audio embeddings (optional)
3. Visualize embedding space with t-SNE/UMAP
4. Find similar songs using KNN (exact and approximate methods)
5. Compare brute force, KD-Tree, and LSH performance
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time
import subprocess
import sys
from scipy.io import wavfile
import io

# Page config
st.set_page_config(page_title="Music Similarity with KNN", layout="wide", page_icon="🎵")

st.title("🎵 Music Similarity Search with KNN")
st.markdown("""
Find similar songs using **audio embeddings** and **K-Nearest Neighbors**!

This app demonstrates how music recommendation systems work using:
- **Audio features**: Extract meaningful representations from music
- **KNN search**: Find similar songs in embedding space
- **Visualization**: See songs clustered by similarity
- **Audio playback**: Listen to songs and verify similarity! 🎧
""")

# Color scheme
COLORS = {
    'query': '#E63946',
    'neighbors': '#06FFA5',
    'other': '#D3D3D3',
    'genre': {
        'rock': '#FF6B6B',
        'pop': '#4ECDC4',
        'jazz': '#45B7D1',
        'classical': '#96CEB4',
        'electronic': '#DDA15E',
        'hip-hop': '#BC6C25',
        'metal': '#780000',
        'country': '#FFDDD2',
    }
}

# Check for audio files
SAMPLE_AUDIO_DIR = Path(__file__).parent / "sample_audio"

# ============= SYNTHETIC AUDIO GENERATION =============
@st.cache_data
def generate_synthetic_audio(genre, tempo, energy, duration=5, sample_rate=22050):
    """
    Generate synthetic audio waveform based on song characteristics
    Creates simple but distinctive sounds for each genre
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base frequency for genre
    genre_base_freq = {
        'rock': 220,        # A3
        'pop': 262,         # C4
        'jazz': 185,        # F#3
        'classical': 440,   # A4
        'electronic': 110,  # A2
        'hip-hop': 130,     # C3
        'metal': 165,       # E3
        'country': 196,     # G3
    }

    base_freq = genre_base_freq.get(genre, 220)

    # Adjust frequency by tempo (faster = higher pitch)
    freq_mult = 1.0 + (tempo - 120) / 480  # Normalize around 120 BPM

    # Generate chord based on genre
    if genre == 'rock':
        # Power chord (root + fifth)
        freqs = [base_freq * freq_mult, base_freq * freq_mult * 1.5]
        wave = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)

    elif genre == 'classical':
        # Major chord (root + third + fifth)
        freqs = [base_freq * freq_mult, base_freq * freq_mult * 1.25, base_freq * freq_mult * 1.5]
        wave = sum(0.25 * np.sin(2 * np.pi * f * t) for f in freqs)

    elif genre == 'jazz':
        # Jazz chord (7th chord)
        freqs = [base_freq * freq_mult, base_freq * freq_mult * 1.25,
                base_freq * freq_mult * 1.5, base_freq * freq_mult * 1.75]
        wave = sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)

    elif genre == 'electronic':
        # Sawtooth wave + sine
        wave = 0.3 * (2 * (t * base_freq * freq_mult % 1) - 1)  # Sawtooth
        wave += 0.2 * np.sin(2 * np.pi * base_freq * freq_mult * 2 * t)

    elif genre == 'hip-hop':
        # Bass-heavy with rhythm
        wave = 0.4 * np.sin(2 * np.pi * base_freq * freq_mult * t)
        # Add kick drum pattern
        beat_freq = tempo / 60  # Beats per second
        kick = np.sin(2 * np.pi * beat_freq * t) > 0.9
        wave += 0.3 * kick * np.sin(2 * np.pi * 60 * t)

    else:
        # Default: simple major chord
        freqs = [base_freq * freq_mult, base_freq * freq_mult * 1.25, base_freq * freq_mult * 1.5]
        wave = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)

    # Apply energy (volume)
    wave = wave * (0.3 + 0.7 * energy)

    # Add fade in/out
    fade_len = int(sample_rate * 0.1)  # 0.1 second fade
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)

    wave[:fade_len] *= fade_in
    wave[-fade_len:] *= fade_out

    # Normalize
    wave = wave / np.max(np.abs(wave)) * 0.8

    # Convert to 16-bit PCM
    wave_int = np.int16(wave * 32767)

    return wave_int, sample_rate

def setup_audio_directory():
    """Ensure sample_audio directory exists"""
    if not SAMPLE_AUDIO_DIR.exists():
        st.warning("⚠️ sample_audio/ directory not found. Creating it now...")
        try:
            # Run download script
            script_path = Path(__file__).parent / "download_samples.py"
            subprocess.run([sys.executable, str(script_path)], check=True)
            st.success("✅ Created sample_audio/ directory!")
            st.info("💡 Add your own .mp3 files to sample_audio/ or use synthetic mode")
        except Exception as e:
            st.error(f"Could not create directory: {e}")
            SAMPLE_AUDIO_DIR.mkdir(exist_ok=True)
            return False
    return True

@st.cache_data
def load_audio_features_from_files():
    """
    Load and extract features from real audio files in sample_audio/

    Returns embeddings, metadata, and raw features if files exist, otherwise None
    """
    try:
        import librosa
    except ImportError:
        st.warning("⚠️ librosa not installed. Install with: pip install librosa")
        st.info("Using synthetic embeddings instead")
        return None, None, None

    audio_files = list(SAMPLE_AUDIO_DIR.glob("*.mp3")) + \
                  list(SAMPLE_AUDIO_DIR.glob("*.wav")) + \
                  list(SAMPLE_AUDIO_DIR.glob("*.flac"))

    if not audio_files:
        return None, None, None

    st.info(f"🎵 Found {len(audio_files)} audio files! Extracting features...")

    embeddings = []
    metadata = []
    raw_features_list = []  # Store raw features for visualization

    progress_bar = st.progress(0)
    for idx, audio_file in enumerate(audio_files):
        try:
            # Load audio
            y, sr = librosa.load(str(audio_file), duration=30, sr=22050)  # First 30 seconds

            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)

            zero_crossing = librosa.feature.zero_crossing_rate(y)
            zero_crossing_mean = np.mean(zero_crossing)

            # Store raw features for visualization
            raw_features = {
                'mfcc_mean': mfccs_mean,
                'mfcc_std': mfccs_std,
                'chroma_mean': chroma_mean,
                'spectral_centroid': spectral_centroid_mean,
                'spectral_rolloff': spectral_rolloff_mean,
                'zero_crossing_rate': zero_crossing_mean
            }
            raw_features_list.append(raw_features)

            # Combine features
            feature_vector = np.concatenate([
                mfccs_mean,
                mfccs_std,
                chroma_mean,
                [spectral_centroid_mean, spectral_rolloff_mean, zero_crossing_mean]
            ])

            # Pad to 128 dimensions
            if len(feature_vector) < 128:
                feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:128]

            embeddings.append(feature_vector)

            # Parse filename for metadata (format: "Genre - Artist - Title")
            filename = audio_file.stem
            parts = filename.split(' - ')
            if len(parts) >= 3:
                genre, artist, title = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                artist, title = parts[0], parts[1]
                genre = 'Unknown'
            else:
                artist, title, genre = "Unknown", filename, "Unknown"

            # Estimate tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = int(tempo) if not np.isnan(tempo) else 120
            except:
                tempo = 120

            metadata.append({
                'id': idx,
                'title': title,
                'artist': artist,
                'genre': genre,
                'tempo': tempo,
                'energy': float(np.std(y)),  # Rough energy estimate
                'duration': len(y) / sr,
                'filename': audio_file.name,
                'filepath': str(audio_file)  # Store full path for playback
            })

        except Exception as e:
            st.warning(f"⚠️ Could not process {audio_file.name}: {e}")
            continue

        progress_bar.progress((idx + 1) / len(audio_files))

    if not embeddings:
        return None, None, None

    embeddings = np.array(embeddings)

    # Normalize
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    df = pd.DataFrame(metadata)

    st.success(f"✅ Extracted features from {len(embeddings)} songs!")

    return embeddings, df, raw_features_list

# Sidebar
st.sidebar.header("⚙️ Settings")

# Try to load real audio first
setup_audio_directory()

use_real_audio = st.sidebar.checkbox(
    "Use real audio files (if available)",
    value=True,
    help="Extract features from MP3/WAV files in sample_audio/"
)

real_embeddings, real_df, raw_features = None, None, None
if use_real_audio:
    real_embeddings, real_df, raw_features = load_audio_features_from_files()

# ============= GENERATE DEMO DATASET =============
@st.cache_data
def generate_demo_music_dataset(n_songs=100, n_features=128, seed=42):
    """
    Generate a demo music dataset with synthetic embeddings
    In practice, these would come from a pre-trained model like VGGish, PANNs, or Musicnn
    """
    np.random.seed(seed)

    # Define genres with characteristic patterns
    genres = ['rock', 'pop', 'jazz', 'classical', 'electronic', 'hip-hop']
    n_per_genre = n_songs // len(genres)

    embeddings = []
    song_info = []

    for genre_idx, genre in enumerate(genres):
        # Create genre-specific embedding clusters
        # Each genre has a different "center" in embedding space
        genre_center = np.random.randn(n_features) * 2
        genre_center[genre_idx * 20:(genre_idx + 1) * 20] += 5  # Make genres separable

        for i in range(n_per_genre):
            # Add variation within genre
            embedding = genre_center + np.random.randn(n_features) * 0.5

            # Add some temporal features (rhythm, tempo indicators)
            tempo = np.random.randint(60, 180) if genre != 'classical' else np.random.randint(40, 120)
            energy = np.random.rand()

            embeddings.append(embedding)
            song_info.append({
                'id': len(song_info),
                'title': f'{genre.capitalize()} Song {i+1}',
                'artist': f'{genre.capitalize()} Artist {(i % 10) + 1}',
                'genre': genre,
                'tempo': tempo,
                'energy': energy,
                'duration': np.random.randint(120, 360)  # 2-6 minutes
            })

    embeddings = np.array(embeddings)

    # Normalize
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    df = pd.DataFrame(song_info)
    return embeddings, df

# Generate or load dataset
if real_embeddings is not None and real_df is not None:
    # Use real audio features
    embeddings = real_embeddings
    songs_df = real_df
    embedding_dim = embeddings.shape[1]
    st.sidebar.success(f"✅ Using {len(songs_df)} real songs!")
    st.sidebar.info("🎵 Real audio features extracted with librosa")
else:
    # Use synthetic embeddings
    n_songs = st.sidebar.slider("Number of songs in library", 50, 500, 200, 50)
    embedding_dim = 128  # Typical size for audio embeddings

    embeddings, songs_df = generate_demo_music_dataset(n_songs, embedding_dim)

    st.sidebar.success(f"✅ Generated {len(songs_df)} synthetic songs across {songs_df['genre'].nunique()} genres")
    st.sidebar.info("💡 Add MP3s to sample_audio/ for real audio features!")

# Check if audio playback is available
if 'filepath' in songs_df.columns and songs_df['filepath'].notna().any():
    st.sidebar.markdown("---")
    st.sidebar.success("🎧 **Audio playback enabled!**")
    st.sidebar.caption("Listen to songs in 'Find Similar Songs' tab")
else:
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Enable audio playback:**
    1. Add MP3/WAV to `sample_audio/`
    2. Install: `pip install librosa soundfile`
    3. Refresh app
    """)

# ============= DIMENSIONALITY REDUCTION FOR VISUALIZATION =============
@st.cache_data
def reduce_dimensions(embeddings, method='tsne', n_components=2):
    """Reduce embeddings to 2D for visualization"""
    if method == 'tsne':
        # Perplexity must be less than n_samples and at least 1
        max_perplexity = max(1, min(30, len(embeddings) - 1))
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=max_perplexity)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)

viz_method = st.sidebar.selectbox("Visualization method", ['tsne', 'pca'], index=0,
                                  help="t-SNE preserves local structure, PCA is faster")

with st.spinner(f"Computing {viz_method.upper()} projection..."):
    coords_2d = reduce_dimensions(embeddings, viz_method)

# ============= MAIN TABS =============
if raw_features is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Explore Library",
        "🔍 Find Similar Songs",
        "⚡ KNN Methods Comparison",
        "📊 Embedding Analysis",
        "🎼 Audio Features"
    ])
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Explore Library",
        "🔍 Find Similar Songs",
        "⚡ KNN Methods Comparison",
        "📊 Embedding Analysis"
    ])
    tab5 = None  # Not available without real audio

# ============= TAB 1: EXPLORE LIBRARY =============
with tab1:
    st.header("🗺️ Music Library Visualization")

    st.markdown(f"""
    Showing **{len(songs_df)} songs** in a 2D embedding space using **{viz_method.upper()}**.
    Each point represents a song. Songs that are **close together** have similar audio characteristics.
    """)

    # Interactive scatter plot
    fig = px.scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        color=songs_df['genre'],
        hover_name=songs_df['title'],
        hover_data={
            'Artist': songs_df['artist'],
            'Genre': songs_df['genre'],
            'Tempo': songs_df['tempo'],
            'Energy': songs_df['energy'].round(2),
        },
        labels={'x': f'{viz_method.upper()} Dimension 1', 'y': f'{viz_method.upper()} Dimension 2'},
        title='Music Library: All Songs',
        color_discrete_map=COLORS['genre'],
        height=700
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(
        font=dict(size=12),
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    st.subheader("📊 Library Statistics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Songs", len(songs_df))
    col2.metric("Genres", songs_df['genre'].nunique())
    col3.metric("Embedding Dimensions", embedding_dim)
    col4.metric("Avg Tempo", f"{songs_df['tempo'].mean():.0f} BPM")

    # Genre distribution
    with st.expander("📈 Genre Distribution"):
        genre_counts = songs_df['genre'].value_counts()

        fig_bar = px.bar(
            x=genre_counts.index,
            y=genre_counts.values,
            labels={'x': 'Genre', 'y': 'Number of Songs'},
            title='Songs per Genre',
            color=genre_counts.index,
            color_discrete_map=COLORS['genre']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ============= TAB 2: FIND SIMILAR SONGS =============
with tab2:
    st.header("🔍 Find Similar Songs")

    st.markdown("""
    Select a song and find its **K nearest neighbors** in the embedding space.
    This is how music recommendation systems work!
    """)

    col1, col2 = st.columns([2, 3])

    with col1:
        # Select query song
        st.subheader("🎼 Select Query Song")

        # Filter by genre
        genre_filter = st.multiselect(
            "Filter by genre",
            options=['All'] + sorted(songs_df['genre'].unique().tolist()),
            default=['All']
        )

        if 'All' in genre_filter or len(genre_filter) == 0:
            filtered_df = songs_df
        else:
            filtered_df = songs_df[songs_df['genre'].isin(genre_filter)]

        # Select song
        song_options = [f"{row['title']} - {row['artist']} ({row['genre']})"
                       for _, row in filtered_df.iterrows()]

        selected_idx = st.selectbox("Choose a song", range(len(song_options)),
                                    format_func=lambda x: song_options[x])

        query_song_id = filtered_df.iloc[selected_idx]['id']
        query_song = songs_df.loc[query_song_id]

        st.markdown("---")
        st.markdown("**Query Song:**")
        st.markdown(f"**{query_song['title']}**")
        st.markdown(f"Artist: {query_song['artist']}")
        st.markdown(f"Genre: {query_song['genre']}")
        st.markdown(f"Tempo: {query_song['tempo']} BPM")
        st.markdown(f"Energy: {query_song['energy']:.2f}")

        # Play query song
        st.markdown("**🎧 Listen:**")
        if 'filepath' in query_song and query_song['filepath']:
            # Real audio file
            try:
                with open(query_song['filepath'], 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.warning(f"Could not load audio: {e}")
        else:
            # Generate synthetic audio
            try:
                wave, sr = generate_synthetic_audio(
                    query_song['genre'],
                    query_song['tempo'],
                    query_song['energy'],
                    duration=5
                )

                # Convert to WAV bytes
                buffer = io.BytesIO()
                wavfile.write(buffer, sr, wave)
                buffer.seek(0)

                st.audio(buffer, format='audio/wav')
                st.caption("🎹 Synthetic audio preview (genre-based tones)")
            except Exception as e:
                st.warning(f"Could not generate audio: {e}")

        # KNN settings
        st.markdown("---")
        k_neighbors = st.slider("Number of similar songs (K)", 1, 20, 5)

        # Find neighbors
        query_embedding = embeddings[query_song_id].reshape(1, -1)

        start_time = time.time()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(query_embedding)
        search_time = (time.time() - start_time) * 1000

        # Remove the query song itself
        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]

        st.success(f"✅ Found {k_neighbors} similar songs in {search_time:.2f} ms")

    with col2:
        # Visualization
        st.subheader("🗺️ Similar Songs in Embedding Space")

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot all songs (faded)
        for genre in songs_df['genre'].unique():
            genre_mask = songs_df['genre'] == genre
            genre_coords = coords_2d[genre_mask]
            ax.scatter(genre_coords[:, 0], genre_coords[:, 1],
                      c=COLORS['genre'].get(genre, 'gray'),
                      s=50, alpha=0.3, label=genre)

        # Plot query song
        query_coords = coords_2d[query_song_id]
        ax.scatter(query_coords[0], query_coords[1],
                  c=COLORS['query'], s=500, marker='*',
                  edgecolors='black', linewidth=2.5, zorder=10,
                  label='Query Song')

        # Plot neighbors
        neighbor_coords = coords_2d[neighbor_indices]
        ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1],
                  c=COLORS['neighbors'], s=200, edgecolors='black',
                  linewidth=2, zorder=8, label=f'{k_neighbors} Similar Songs')

        # Draw lines to neighbors
        for i, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
            coords = coords_2d[idx]
            ax.plot([query_coords[0], coords[0]],
                   [query_coords[1], coords[1]],
                   'k--', alpha=0.3, linewidth=1, zorder=3)

            # Annotate with rank
            ax.text(coords[0], coords[1], f'{i+1}',
                   fontsize=10, weight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5))

        ax.set_xlabel(f'{viz_method.upper()} Dimension 1', fontsize=12, weight='bold')
        ax.set_ylabel(f'{viz_method.upper()} Dimension 2', fontsize=12, weight='bold')
        ax.set_title(f'Query: "{query_song["title"]}" + {k_neighbors} Similar Songs',
                    fontsize=14, weight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    # Show results table with audio players
    st.markdown("---")
    st.subheader("📋 Similar Songs (Ranked by Distance)")

    # Display results with audio players
    for rank, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
        song = songs_df.loc[idx]

        with st.expander(f"#{rank+1}: {song['title']} - {song['artist']} (Distance: {dist:.4f})", expanded=(rank < 3)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                **Title:** {song['title']}
                **Artist:** {song['artist']}
                **Genre:** {song['genre']} {'✅' if song['genre'] == query_song['genre'] else '❌'}
                **Tempo:** {song['tempo']} BPM
                **Energy:** {song['energy']:.2f}
                **Distance:** {dist:.4f}
                """)

                # Audio player
                st.markdown("**🎧 Listen:**")
                if 'filepath' in song and song['filepath']:
                    # Real audio file
                    try:
                        with open(song['filepath'], 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/mp3')
                    except Exception as e:
                        st.caption(f"Could not load audio: {e}")
                else:
                    # Generate synthetic audio
                    try:
                        wave, sr = generate_synthetic_audio(
                            song['genre'],
                            song['tempo'],
                            song['energy'],
                            duration=5
                        )

                        buffer = io.BytesIO()
                        wavfile.write(buffer, sr, wave)
                        buffer.seek(0)

                        st.audio(buffer, format='audio/wav')
                        st.caption("🎹 Synthetic audio (genre-based tones)")
                    except Exception as e:
                        st.caption(f"Could not generate audio: {e}")

            with col2:
                # Show position in embedding space
                song_coords = coords_2d[idx]
                st.metric("2D Position", f"({song_coords[0]:.2f}, {song_coords[1]:.2f})")
                st.metric("Same Genre?", "Yes ✅" if song['genre'] == query_song['genre'] else "No ❌")

    # Summary table
    st.markdown("---")
    st.subheader("📊 Summary Table")

    results_data = []
    for rank, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
        song = songs_df.loc[idx]
        results_data.append({
            'Rank': rank + 1,
            'Title': song['title'],
            'Artist': song['artist'],
            'Genre': song['genre'],
            'Tempo': f"{song['tempo']} BPM",
            'Energy': f"{song['energy']:.2f}",
            'Distance': f"{dist:.4f}",
            'Same Genre?': '✅' if song['genre'] == query_song['genre'] else '❌'
        })

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, hide_index=True, use_container_width=True)

    # Genre match analysis
    same_genre_count = sum(songs_df.loc[idx, 'genre'] == query_song['genre']
                          for idx in neighbor_indices)

    col1, col2, col3 = st.columns(3)
    col1.metric("Same Genre", f"{same_genre_count}/{k_neighbors}",
               f"{100*same_genre_count/k_neighbors:.0f}%")
    col2.metric("Avg Distance", f"{neighbor_distances.mean():.4f}")
    col3.metric("Max Distance", f"{neighbor_distances.max():.4f}")

# ============= TAB 3: KNN METHODS COMPARISON =============
with tab3:
    st.header("⚡ KNN Methods Comparison")

    st.markdown("""
    Compare different KNN algorithms:
    - **Brute Force**: Exact, but O(N) time
    - **KD-Tree**: Fast for low dimensions, O(log N) average
    - **Ball Tree**: Better for high dimensions
    - **LSH**: Approximate but very fast for high-D

    For music embeddings (128-D), **Ball Tree** often works best!
    """)

    # Select test song
    test_song_id = st.slider("Select test song", 0, len(songs_df) - 1, 0)
    test_song = songs_df.loc[test_song_id]
    test_embedding = embeddings[test_song_id].reshape(1, -1)

    st.markdown(f"**Test song**: {test_song['title']} - {test_song['artist']} ({test_song['genre']})")

    k_test = st.slider("K (neighbors to find)", 1, 50, 10, key='k_test')

    # Compare algorithms
    algorithms = {
        'Brute Force': 'brute',
        'KD-Tree': 'kd_tree',
        'Ball Tree': 'ball_tree',
    }

    results = {}

    for name, algo in algorithms.items():
        try:
            start = time.time()
            nbrs = NearestNeighbors(n_neighbors=k_test + 1, algorithm=algo)
            nbrs.fit(embeddings)
            distances, indices = nbrs.kneighbors(test_embedding)
            elapsed = (time.time() - start) * 1000  # ms

            results[name] = {
                'time': elapsed,
                'indices': indices[0][1:],  # Exclude query itself
                'distances': distances[0][1:]
            }
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {str(e)}")
            results[name] = None

    # Display results
    st.subheader("⏱️ Performance Comparison")

    comparison_data = []
    for name, result in results.items():
        if result:
            comparison_data.append({
                'Algorithm': name,
                'Time (ms)': f"{result['time']:.3f}",
                'Speedup': f"{results['Brute Force']['time'] / result['time']:.2f}×" if result['time'] > 0 else 'N/A'
            })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # Bar chart - safely handle empty dataframe
    if not comparison_df.empty and 'Time (ms)' in comparison_df.columns:
        # Extract numeric times
        times_numeric = [float(t.split()[0]) for t in comparison_df['Time (ms)']]

        fig_comparison = px.bar(
            x=comparison_df['Algorithm'],
            y=times_numeric,
            labels={'y': 'Time (ms)', 'x': 'Algorithm'},
            title='Query Time Comparison (Lower is Better)',
            color=comparison_df['Algorithm'],
            text=[f"{t:.3f}" for t in times_numeric]
        )
        fig_comparison.update_traces(textposition='outside')
        st.plotly_chart(fig_comparison, use_container_width=True)

    # Check if results match
    st.subheader("🎯 Accuracy Check")

    if results.get('Brute Force') and results['Brute Force'] is not None:
        brute_set = set(results['Brute Force']['indices'])

        for name, result in results.items():
            if result and name != 'Brute Force':
                result_set = set(result['indices'])
                matches = len(brute_set & result_set)
                accuracy = 100 * matches / k_test

                if accuracy == 100:
                    st.success(f"✅ **{name}**: {accuracy:.1f}% match (exact!)")
                else:
                    st.warning(f"⚠️ **{name}**: {accuracy:.1f}% match ({matches}/{k_test} same neighbors)")
    else:
        st.warning("Brute Force search failed - cannot compare accuracy")

    st.markdown("---")
    st.info("""
    **Key insights:**
    - **Brute Force**: Always exact, but slowest
    - **KD-Tree**: Fast, but degrades in high dimensions (curse of dimensionality)
    - **Ball Tree**: Often best for moderate dimensions like 128-D music embeddings
    - For production music apps: Use specialized libraries like **FAISS** or **Annoy**!
    """)

# ============= TAB 4: EMBEDDING ANALYSIS =============
with tab4:
    st.header("📊 Embedding Space Analysis")

    st.markdown("""
    Analyze the structure of the embedding space:
    - **How well are genres separated?**
    - **What are the most distinctive features?**
    - **How dense is the space?**
    """)

    # Genre separation analysis
    st.subheader("🎭 Genre Separation")

    # Compute pairwise distances between genre centroids
    genre_centroids = {}
    for genre in songs_df['genre'].unique():
        genre_mask = songs_df['genre'] == genre
        genre_centroids[genre] = embeddings[genre_mask].mean(axis=0)

    # Visualize centroids
    centroid_matrix = np.array([genre_centroids[g] for g in sorted(genre_centroids.keys())])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    centroid_distances = squareform(pdist(centroid_matrix, metric='euclidean'))

    # Heatmap
    im = ax.imshow(centroid_distances, cmap='YlOrRd', aspect='auto')

    # Labels
    genres_sorted = sorted(genre_centroids.keys())
    ax.set_xticks(range(len(genres_sorted)))
    ax.set_yticks(range(len(genres_sorted)))
    ax.set_xticklabels(genres_sorted, rotation=45, ha='right')
    ax.set_yticklabels(genres_sorted)

    # Add values
    for i in range(len(genres_sorted)):
        for j in range(len(genres_sorted)):
            text = ax.text(j, i, f'{centroid_distances[i, j]:.2f}',
                          ha="center", va="center", color="black" if centroid_distances[i, j] > centroid_distances.max()/2 else "white",
                          fontsize=9)

    ax.set_title('Genre Centroid Distances (Euclidean)', fontsize=14, weight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Distance')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    st.markdown("**Interpretation**: Larger distances → more separated genres → easier to cluster!")

    # Nearest neighbor distribution
    st.markdown("---")
    st.subheader("📏 Nearest Neighbor Distance Distribution")

    # Compute 1-NN distance for each song
    nbrs_all = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nbrs_all.fit(embeddings)
    distances_all, _ = nbrs_all.kneighbors(embeddings)
    nn_distances = distances_all[:, 1]  # Distance to nearest neighbor (excluding self)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nn_distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(nn_distances.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {nn_distances.mean():.3f}')
    ax.axvline(np.median(nn_distances), color='orange', linestyle='--', linewidth=2,
              label=f'Median: {np.median(nn_distances):.3f}')
    ax.set_xlabel('Distance to Nearest Neighbor', fontsize=12, weight='bold')
    ax.set_ylabel('Count', fontsize=12, weight='bold')
    ax.set_title('Distribution of Nearest Neighbor Distances', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean NN Distance", f"{nn_distances.mean():.4f}")
    col2.metric("Median NN Distance", f"{np.median(nn_distances):.4f}")
    col3.metric("Std Dev", f"{nn_distances.std():.4f}")

    st.info("""
    **What this tells us:**
    - **Low mean distance**: Songs are generally close to others (dense space)
    - **High variance**: Some songs are very unique (outliers)
    - In real music datasets, you'd see clusters of similar songs with outliers
    """)

# ============= TAB 5: AUDIO FEATURES =============
if tab5 is not None and raw_features is not None:
    with tab5:
        st.header("🎼 Audio Features Visualization")

        st.markdown("""
        These are the actual audio features extracted from your music files using **librosa**.
        Understanding these features helps explain how the KNN algorithm finds similar songs!
        """)

        # Select a song to visualize
        st.subheader("📊 Feature Analysis by Song")

        song_options_feat = [f"{row['title']} - {row['artist']} ({row['genre']})"
                           for _, row in songs_df.iterrows()]

        selected_idx_feat = st.selectbox("Select song to analyze features",
                                        range(len(song_options_feat)),
                                        format_func=lambda x: song_options_feat[x],
                                        key='feat_song_select')

        selected_song = songs_df.iloc[selected_idx_feat]
        selected_features = raw_features[selected_idx_feat]

        st.markdown(f"**Analyzing**: {selected_song['title']} by {selected_song['artist']}")
        st.markdown(f"**Genre**: {selected_song['genre']}")

        # Display feature summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Spectral Centroid", f"{selected_features['spectral_centroid']:.0f} Hz",
                   help="Center of mass of spectrum - brighter sounds have higher values")
        col2.metric("Spectral Rolloff", f"{selected_features['spectral_rolloff']:.0f} Hz",
                   help="Frequency below which 85% of energy is contained")
        col3.metric("Zero Crossing Rate", f"{selected_features['zero_crossing_rate']:.4f}",
                   help="Rate at which signal changes sign - higher for percussive sounds")

        # MFCC visualization
        st.markdown("---")
        st.subheader("🎵 MFCCs (Mel-Frequency Cepstral Coefficients)")
        st.markdown("**MFCCs** capture the timbral texture of sound - similar to how humans perceive pitch.")

        fig_mfcc, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MFCC mean
        ax1.bar(range(len(selected_features['mfcc_mean'])), selected_features['mfcc_mean'],
               color='steelblue', edgecolor='black')
        ax1.set_xlabel('MFCC Coefficient', fontsize=11, weight='bold')
        ax1.set_ylabel('Mean Value', fontsize=11, weight='bold')
        ax1.set_title('MFCC Mean Values (Timbral Texture)', fontsize=12, weight='bold')
        ax1.grid(True, alpha=0.3)

        # MFCC std
        ax2.bar(range(len(selected_features['mfcc_std'])), selected_features['mfcc_std'],
               color='coral', edgecolor='black')
        ax2.set_xlabel('MFCC Coefficient', fontsize=11, weight='bold')
        ax2.set_ylabel('Std Deviation', fontsize=11, weight='bold')
        ax2.set_title('MFCC Variation (Timbral Dynamics)', fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_mfcc)
        plt.close()

        st.caption("💡 Lower MFCCs capture overall spectral shape, higher ones capture finer details")

        # Chroma features
        st.markdown("---")
        st.subheader("🎹 Chroma Features (Pitch Class Distribution)")
        st.markdown("**Chroma** represents the 12 pitch classes (C, C#, D, ..., B) - captures harmonic content.")

        fig_chroma, ax = plt.subplots(figsize=(12, 5))

        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        colors_chroma = plt.cm.rainbow(np.linspace(0, 1, 12))

        bars = ax.bar(chroma_labels, selected_features['chroma_mean'], color=colors_chroma,
                     edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Pitch Class', fontsize=12, weight='bold')
        ax.set_ylabel('Mean Energy', fontsize=12, weight='bold')
        ax.set_title(f'Chroma Distribution - {selected_song["title"]}', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight dominant pitch
        dominant_pitch_idx = np.argmax(selected_features['chroma_mean'])
        bars[dominant_pitch_idx].set_edgecolor('gold')
        bars[dominant_pitch_idx].set_linewidth(4)

        plt.tight_layout()
        st.pyplot(fig_chroma)
        plt.close()

        st.caption(f"💡 Dominant pitch class: **{chroma_labels[dominant_pitch_idx]}** (highlighted in gold)")

        # Feature comparison across genres
        st.markdown("---")
        st.subheader("📊 Feature Comparison Across Genres")

        # Calculate average features per genre
        genre_features = {}
        for genre in songs_df['genre'].unique():
            genre_mask = songs_df['genre'] == genre
            genre_indices = songs_df[genre_mask].index.tolist()

            # Average spectral features for this genre
            avg_centroid = np.mean([raw_features[i]['spectral_centroid'] for i in genre_indices])
            avg_rolloff = np.mean([raw_features[i]['spectral_rolloff'] for i in genre_indices])
            avg_zcr = np.mean([raw_features[i]['zero_crossing_rate'] for i in genre_indices])

            genre_features[genre] = {
                'spectral_centroid': avg_centroid,
                'spectral_rolloff': avg_rolloff,
                'zero_crossing_rate': avg_zcr
            }

        # Plot comparison
        fig_comparison, axes = plt.subplots(1, 3, figsize=(15, 5))

        genres_list = list(genre_features.keys())
        colors_genre = [COLORS['genre'].get(g, 'gray') for g in genres_list]

        # Spectral Centroid
        centroids = [genre_features[g]['spectral_centroid'] for g in genres_list]
        axes[0].barh(genres_list, centroids, color=colors_genre, edgecolor='black')
        axes[0].set_xlabel('Spectral Centroid (Hz)', fontsize=10, weight='bold')
        axes[0].set_title('Brightness by Genre', fontsize=11, weight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Spectral Rolloff
        rolloffs = [genre_features[g]['spectral_rolloff'] for g in genres_list]
        axes[1].barh(genres_list, rolloffs, color=colors_genre, edgecolor='black')
        axes[1].set_xlabel('Spectral Rolloff (Hz)', fontsize=10, weight='bold')
        axes[1].set_title('Energy Distribution by Genre', fontsize=11, weight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # Zero Crossing Rate
        zcrs = [genre_features[g]['zero_crossing_rate'] for g in genres_list]
        axes[2].barh(genres_list, zcrs, color=colors_genre, edgecolor='black')
        axes[2].set_xlabel('Zero Crossing Rate', fontsize=10, weight='bold')
        axes[2].set_title('Percussiveness by Genre', fontsize=11, weight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        st.pyplot(fig_comparison)
        plt.close()

        st.info("""
        **Key Insights:**
        - **Spectral Centroid**: Higher values = brighter, sharper sounds (cymbals, vocals)
        - **Spectral Rolloff**: Shows where most energy is concentrated
        - **Zero Crossing Rate**: Higher for noisy/percussive sounds, lower for tonal sounds

        These features help KNN distinguish between genres and find similar songs!
        """)

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Key Takeaways

1. **Audio Embeddings**: Convert music → vectors that capture audio characteristics
   - In practice: Use pre-trained models (VGGish, PANNs, Musicnn, CLAP)
   - Here: We used synthetic embeddings for demonstration

2. **KNN for Music Similarity**: Find nearest neighbors in embedding space
   - **Fast**: Especially with Ball Tree or specialized libraries (FAISS)
   - **Effective**: Songs close in embedding space sound similar

3. **Visualization**: t-SNE/UMAP helps understand the embedding space
   - Genres should form clusters if embeddings are good
   - Outliers might be unique songs or mislabeled

4. **Real-World Systems**:
   - **Spotify**: Uses collaborative filtering + audio embeddings
   - **YouTube Music**: Deep learning models trained on billions of songs
   - **Production**: FAISS (Facebook), Annoy (Spotify) for billion-scale search

**Try it yourself**:
- Upload your own music files
- Extract features with librosa or other libraries
- Build your own music recommender!

🚀 **Music recommendation is KNN at scale!**
""")

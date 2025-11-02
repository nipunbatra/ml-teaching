## 🎵 Music Similarity Search with KNN

Find similar songs using audio embeddings and K-Nearest Neighbors - just like Spotify!

![Music](https://img.shields.io/badge/Topic-Music%20Recommendation-purple)
![KNN](https://img.shields.io/badge/Algorithm-KNN-blue)
![Embeddings](https://img.shields.io/badge/Features-Audio%20Embeddings-green)

## 🎯 What You'll Learn

1. **Audio Embeddings**: How to represent music as vectors
2. **KNN for Recommendations**: Finding similar songs in embedding space
3. **Algorithm Comparison**: Brute force vs KD-Tree vs Ball Tree
4. **Visualization**: Understanding embedding spaces with t-SNE/UMAP
5. **Real-World Applications**: How music streaming services work

## ✨ Features

### 🗺️ Tab 1: Explore Library
- Visualize entire music library in 2D (t-SNE or PCA)
- See how songs cluster by genre
- Interactive exploration with hover details
- Genre distribution analysis

### 🔍 Tab 2: Find Similar Songs
- Select any song as query
- **🎧 Listen to EVERY song** - both real MP3s AND synthetic audio!
  - Real audio: Play actual MP3/WAV files
  - Synthetic audio: Genre-based tones/chords (rock, jazz, classical, etc.)
- Find K nearest neighbors instantly
- Visualize query + neighbors in embedding space
- Detailed results with genre matching analysis
- Expandable cards with audio players for each result
- **Works perfectly in demo mode** - no MP3s required!

### ⚡ Tab 3: KNN Methods Comparison
- Compare Brute Force, KD-Tree, Ball Tree
- Measure query time for each algorithm
- Verify accuracy of approximate methods
- Understand which algorithm works best for high-D data

### 📊 Tab 4: Embedding Analysis
- Genre separation heatmap
- Nearest neighbor distance distribution
- Quality metrics for embeddings
- Insights into embedding space structure

## 🚀 Quick Start

### Option 1: Demo Mode (No Audio Files Needed)
```bash
cd apps/music-similarity

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
The app works perfectly with synthetic embeddings - no MP3 files required!

### Option 2: With Real Audio Playback 🎧
```bash
cd apps/music-similarity

# Install dependencies including librosa
pip install -r requirements.txt
pip install librosa soundfile

# Add your music files
# Copy .mp3 or .wav files to sample_audio/ directory
# Name them: "Artist - Song Title.mp3"

# Run the app
streamlit run app.py
```

The app will auto-detect audio files and enable playback! 🎵

### Adding Your Music Files

**File Naming Convention:**
```
Artist - Song Title.mp3       ✅ Good
ArtistName - SongName.wav     ✅ Good
song.mp3                      ⚠️ Works but metadata limited
```

**Supported Formats:**
- `.mp3` (most common)
- `.wav` (uncompressed)
- `.flac` (lossless)
- `.ogg` (Vorbis)

**Example:**
```bash
cd apps/music-similarity/sample_audio/

# Add your files
cp ~/Music/Beatles\ -\ Hey\ Jude.mp3 .
cp ~/Music/Queen\ -\ Bohemian\ Rhapsody.mp3 .
# ... add more songs

# Run the app
cd ..
streamlit run app.py
```

The app will extract features and you can play each song! 🎧

## 🎓 Educational Use

### For Teaching Music Information Retrieval

**Lecture 1: Audio Embeddings (30 min)**
1. What are embeddings? (5 min)
   - Motivation: Need numerical representation of audio
   - Show how similar songs map to nearby vectors

2. Explore Library Tab (10 min)
   - Show t-SNE visualization
   - Point out genre clusters
   - Discuss what makes songs similar

3. Real-World Embeddings (15 min)
   - MFCCs, spectral features (traditional)
   - Deep learning: VGGish, PANNs, Musicnn
   - Modern: CLAP (contrastive learning)

**Lecture 2: KNN for Recommendations (30 min)**
1. Find Similar Songs Tab (10 min)
   - Demo the search interface
   - Show how distance correlates with similarity
   - Discuss genre vs acoustic similarity

2. Algorithm Comparison (15 min)
   - Compare brute force vs tree-based methods
   - Explain curse of dimensionality
   - Show why Ball Tree works well for 128-D

3. Production Systems (5 min)
   - FAISS for billion-scale search
   - Spotify's use of Annoy
   - YouTube Music architecture

### Interactive Exercises

1. **Genre Exploration**
   - Find a rock song, check if neighbors are rock
   - Try with classical → likely all classical neighbors
   - Find "crossover" songs at genre boundaries

2. **Parameter Tuning**
   - Try K=1 vs K=20, how do results differ?
   - Use t-SNE vs PCA, which shows better clusters?

3. **Algorithm Performance**
   - Measure speedup for different methods
   - Try with 50 vs 500 songs, observe scaling

## 🔬 Technical Details

### Synthetic Demo Data

For educational purposes, this app generates synthetic embeddings with:
- **Dimensionality**: 128 (typical for audio embeddings)
- **Genre clusters**: Each genre has distinct center in embedding space
- **Intra-genre variation**: Songs within genre have added noise
- **Metadata**: Tempo, energy, duration for realism

### Real-World Embeddings

To extend this to real music, you could use:

**Traditional Features** (with librosa):
```python
import librosa
y, sr = librosa.load('song.mp3')
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
embedding = np.mean(mfccs, axis=1)  # Time-average
```

**Pre-trained Models**:
- **VGGish**: 128-D embeddings from AudioSet
- **PANNs**: High-quality audio tagging embeddings
- **Musicnn**: Music-specific CNN embeddings
- **CLAP**: Contrastive Language-Audio Pretraining

**Spotify API** (actual Spotify features):
```python
import spotipy
features = sp.audio_features(track_id)
# Get: danceability, energy, valence, tempo, etc.
```

### KNN Algorithms Explained

**Brute Force**:
- Time: O(N·D) where N=songs, D=dimensions
- Always exact, simple
- Too slow for large N

**KD-Tree**:
- Time: O(log N) average, O(N) worst
- Works well for D < 20
- Degrades badly for D=128 (curse of dimensionality)

**Ball Tree**:
- Time: O(log N) more robust than KD-Tree
- Better for moderate D (20-100)
- Recommended for music embeddings

**Production (FAISS/Annoy)**:
- Approximate but very fast
- Can handle billions of songs
- Used by Spotify, YouTube

## 📊 Visualization Methods

**t-SNE** (t-Distributed Stochastic Neighbor Embedding):
- Preserves local structure
- Great for finding clusters
- Non-linear, slow for large N
- Good for exploration

**PCA** (Principal Component Analysis):
- Linear projection
- Fast
- Shows global variance
- Less interpretable for clusters

## 🌍 Real-World Applications

### Music Streaming Services

**Spotify**:
- Combines collaborative filtering + audio embeddings
- Annoy library for fast ANN search
- Factors: listening history + acoustic similarity

**YouTube Music**:
- Deep learning embeddings
- Trained on billions of songs
- Multi-modal: audio + metadata + user behavior

**SoundCloud**:
- Audio fingerprinting for duplicates
- Embedding-based recommendations
- Genre-aware search

### Other Domains Using Similar Approaches

- **Image search**: Visual embeddings + KNN
- **Document retrieval**: Text embeddings + ANN
- **Product recommendations**: Item embeddings
- **Drug discovery**: Molecular embeddings

## 🔧 Extensions & Projects

### Beginner Projects

1. **Add Real Audio**:
   - Use librosa to extract MFCC features
   - Load .mp3/.wav files
   - Build embeddings from scratch

2. **Genre Classifier**:
   - Use embeddings as features
   - Train classifier (KNN, SVM, neural net)
   - Evaluate accuracy

3. **Playlist Generator**:
   - Start with seed song
   - Iteratively add nearest neighbors
   - Create "radio" feature

### Advanced Projects

1. **Deep Embeddings**:
   - Fine-tune VGGish or PANNs
   - Train on your own music collection
   - Compare with hand-crafted features

2. **Multi-Modal**:
   - Combine audio + lyrics + metadata
   - Fusion strategies
   - Improve recommendations

3. **Interactive Demo**:
   - Upload audio files
   - Real-time feature extraction
   - Live similarity search

4. **Scalability**:
   - Integrate FAISS
   - Handle 1M+ songs
   - Benchmark query times

## 📚 Resources

### Papers

- **VGGish**: "CNN Architectures for Large-Scale Audio Classification" (Google, 2017)
- **PANNs**: "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" (2019)
- **CLAP**: "Large-Scale Contrastive Language-Audio Pretraining" (2022)

### Libraries

- **librosa**: Audio feature extraction
- **essentia**: Comprehensive audio analysis
- **torch-audiomentations**: Data augmentation
- **FAISS**: Billion-scale similarity search
- **Annoy**: Approximate nearest neighbors (Spotify)

### Datasets

- **GTZAN**: Genre classification (1000 songs, 10 genres)
- **Million Song Dataset**: 1M songs with metadata
- **AudioSet**: 2M YouTube clips, 600+ classes
- **Free Music Archive (FMA)**: 106K tracks, metadata

## 🎉 Tips for Best Experience

1. **Start with exploration**: Use Tab 1 to understand the space
2. **Try edge cases**: Find songs at genre boundaries
3. **Compare algorithms**: See which is fastest for your use case
4. **Analyze embeddings**: Check if genres are well-separated
5. **Think about extensions**: How would you add lyrics? User preferences?

## 💡 Discussion Questions

1. **Acoustic vs Semantic Similarity**:
   - Should "similar" mean similar sound or similar meaning?
   - How does genre affect similarity?

2. **Cold Start Problem**:
   - How to recommend new songs with no listening history?
   - Pure audio embeddings vs collaborative filtering?

3. **Diversity vs Accuracy**:
   - Should recommendations be all very similar?
   - How to balance exploration vs exploitation?

4. **Evaluation**:
   - How to measure recommendation quality?
   - User satisfaction vs accuracy?

---

**Built for teaching ML in music!** 🎵🤖

Enjoy exploring music similarity with KNN! 🚀

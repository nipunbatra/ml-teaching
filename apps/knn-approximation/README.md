# K-Nearest Neighbors: Exact vs Approximate Methods

Two complementary interactive educational apps for teaching KNN approximation algorithms.

![KNN Approximation](https://img.shields.io/badge/Topic-KNN%20Approximation-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

## 📱 Two Apps Available

### 🎬 App 1: Step-by-Step Algorithm Walkthrough (`app_stepbystep.py`)
**Best for**: Deep understanding of algorithm mechanics, lecture demonstrations

Watch algorithms come to life! Step through each operation:
- **KD-Tree Construction**: See each split being made, one at a time
- **KD-Tree Query**: Follow the search path, node by node, with backtracking
- **LSH Construction**: Watch hyperplanes being generated and hash tables being built
- **LSH Query**: See hashing, bucket lookup, and candidate retrieval in action

**Use this when**: Teaching the algorithms for the first time, need detailed walkthroughs

### 🔍 App 2: Comparison & Performance Analysis (`app.py`)
**Best for**: Understanding trade-offs, performance comparison, practical applications

Compare all methods side-by-side:
- **Brute Force**: Baseline with 100% accuracy
- **KD-Tree**: Fast exact search with edge cases explained
- **LSH**: Approximate search for high dimensions
- **Performance metrics**: Speed, accuracy, scaling analysis

**Use this when**: Students already understand basics, want to compare methods, discuss trade-offs

## 🎯 Learning Objectives

After using this app, students will understand:

1. **Why approximate methods matter**: Brute force is O(N) - too slow for large datasets
2. **KD-Trees**: Binary space partitioning for fast nearest neighbor search
3. **LSH**: Locality-sensitive hashing for high-dimensional data
4. **Trade-offs**: Speed vs accuracy, when to use each method
5. **Edge cases**: Why KD-trees need backtracking, curse of dimensionality

## 🚀 Features

### Tab 1: Brute Force (Exact)
- Baseline algorithm: compute distance to ALL points
- Time complexity: O(N·D)
- Always 100% accurate
- Visual representation of query radius and neighbors

### Tab 2: KD-Tree
- **Step-by-step visualization**:
  - Shows which nodes are visited vs pruned
  - Displays space partitioning with splitting lines
  - Highlights the search path through the tree

- **Educational edge cases**:
  - Demonstrates when different subtrees contain closest points
  - Shows backtracking requirement
  - Explains curse of dimensionality

- **Performance metrics**:
  - Build time vs search time
  - Number of points checked vs pruned
  - Speedup compared to brute force

### Tab 3: LSH (Locality Sensitive Hashing)
- **Hash function visualization**:
  - Shows random hyperplanes
  - Displays hash values (binary and decimal)
  - Color-coded buckets

- **Search process**:
  - Highlights candidates from all tables
  - Compares LSH results vs true neighbors
  - Shows recall/accuracy

- **Parameter tuning**:
  - Adjustable number of hash tables
  - Adjustable hash bits per table
  - Real-time impact on speed and accuracy

### Tab 4: Comparison & Complexity
- **Side-by-side comparison**: All three methods with metrics
- **Time complexity analysis**: Big-O notation explained
- **Scaling curves**: How each method scales with dataset size
- **Decision guide**: Which method to use when
- **Real-world applications**: Practical use cases

## 🎨 Visualizations

- **Color-coded elements**:
  - 🔴 Red: Query point
  - 🟦 Teal: True nearest neighbors
  - 🟨 Yellow: KD-tree splitting planes
  - 🟧 Orange: Visited nodes
  - ❌ Red X: Pruned nodes
  - 🟩 Light green: LSH candidates

- **Interactive controls**:
  - Adjust query point position
  - Change K (number of neighbors)
  - Modify dataset size and random seed
  - Tune LSH parameters

## 📚 Educational Content

### Key Concepts Covered

1. **KD-Tree Backtracking**:
   - Why checking only one branch isn't enough
   - Hypersphere crossing splitting plane
   - Visual demonstration of the edge case

2. **Curse of Dimensionality**:
   - Why KD-trees fail in high dimensions (D > 20)
   - Theoretical analysis: O(log N) → O(N) degradation
   - When to switch to LSH

3. **LSH Trade-offs**:
   - More tables → higher recall, slower query
   - More bits → finer buckets, might miss neighbors
   - Parameter tuning guide with examples

4. **Time Complexity**:
   - Detailed analysis for each method
   - Average vs worst case
   - Space complexity considerations

5. **Real-World Applications**:
   - Image search (Google Images)
   - Recommendation systems
   - Duplicate detection
   - Robotics & point clouds
   - Database spatial queries

## 🛠️ Installation & Usage

```bash
# Navigate to app directory
cd apps/knn-approximation

# Install dependencies
pip install -r requirements.txt

# Run Step-by-Step App (recommended for first-time teaching)
streamlit run app_stepbystep.py

# OR run Comparison App
streamlit run app.py
```

**Quick start script**:
```bash
# Make executable and run
chmod +x run.sh
./run.sh stepbystep   # For step-by-step app
./run.sh comparison   # For comparison app
```

## 🎓 Usage in Teaching

### Recommended Two-Lecture Approach

**Lecture 1: Algorithm Mechanics (45 min)** - Use `app_stepbystep.py`

1. **KD-Tree Construction** (10 min)
   - Step through tree building, one split at a time
   - Show how median selection works
   - Explain alternating X/Y axes
   - Visualize space partitioning

2. **KD-Tree Query** (15 min)
   - Follow search path node by node
   - **Critical**: Show backtracking example
   - Explain when to prune vs when to search other branch
   - Demonstrate the "search radius crosses split" concept

3. **LSH Construction** (10 min)
   - Generate hyperplanes step by step
   - Hash points one at a time
   - Show bucket formation
   - Explain why similar points hash together

4. **LSH Query** (10 min)
   - Hash query point
   - Lookup buckets in each table
   - Collect candidates
   - Select top K

**Lecture 2: Comparison & Analysis (45 min)** - Use `app.py`

1. **Brute Force Baseline** (5 min)
   - Show exact but slow approach
   - Establish ground truth

2. **KD-Tree Performance** (15 min)
   - Compare speed vs brute force
   - Show accuracy (should be 100%)
   - Experiment with different query positions
   - Discuss when it works well vs poorly

3. **LSH Performance** (15 min)
   - Tune parameters (tables, bits)
   - Observe speed/accuracy trade-off
   - Compare with KD-tree

4. **Real-World Discussion** (10 min)
   - When to use each method
   - Modern implementations (FAISS, HNSW)
   - Hybrid approaches

### Interactive Exercises

1. **Find the edge case**:
   - Move query point to different positions
   - Find a case where KD-tree needs to backtrack
   - Explain why

2. **Parameter tuning**:
   - Start with LSH (2 tables, 2 bits) - poor accuracy
   - Increase tables → observe recall improvement
   - Increase bits → observe candidate reduction

3. **Scaling analysis**:
   - Increase dataset size from 50 to 500
   - Observe speedup changes
   - Discuss theoretical vs empirical complexity

## 📊 Technical Details

### Algorithms Implemented

**KD-Tree Construction**:
```
1. Choose splitting dimension (alternate X, Y)
2. Find median along that dimension
3. Recursively build left and right subtrees
Time: O(N log N)
```

**KD-Tree Search**:
```
1. Traverse tree to find leaf containing query
2. Track best K neighbors so far
3. Backtrack: check if other branches might have closer points
   - If distance to splitting plane < K-th distance: search other branch
   - Otherwise: prune
Time: O(log N) average, O(N) worst case
```

**LSH Construction**:
```
For each table:
  1. Generate random hyperplanes (normal vectors)
  2. For each point:
     - Project onto hyperplanes
     - Create binary hash (+ or -)
     - Store in hash table
Time: O(N · L · B) where L=tables, B=bits
```

**LSH Query**:
```
1. Hash query point for all tables
2. Retrieve candidates from same buckets
3. Compute exact distances to candidates
4. Return top K
Time: O(N^ρ) where ρ ∈ (0.5, 0.8)
```

### Hyperparameters

- **Dataset size**: 50-500 points (adjustable)
- **K**: 1-20 neighbors
- **LSH tables**: 1-10 (recommended: 5)
- **LSH bits**: 2-8 (recommended: 4)

## 🔬 Advanced Topics

### Extensions for Advanced Students

1. **Other LSH families**:
   - Min-Hash for Jaccard similarity
   - p-stable distributions for Lp norms
   - Cross-polytope LSH

2. **Better tree structures**:
   - Ball trees (better for high-D than KD-trees)
   - Cover trees (provable bounds)
   - VP-trees (metric spaces)

3. **Modern methods**:
   - HNSW (Hierarchical Navigable Small World)
   - FAISS (Facebook AI Similarity Search)
   - ScaNN (Google)

4. **Hybrid approaches**:
   - LSH → coarse retrieval (top 1000)
   - Brute force → exact ranking (top 10)
   - Best of both worlds!

## 📖 References

- **KD-Trees**: Bentley, J. L. (1975). "Multidimensional binary search trees"
- **LSH**: Indyk, P., & Motwani, R. (1998). "Approximate nearest neighbors"
- **Curse of dimensionality**: Beyer et al. (1999). "When is nearest neighbor meaningful?"

## 🤝 Contributing

Suggestions for improvements:

- [ ] Add 3D visualization option
- [ ] Implement Ball-tree comparison
- [ ] Add animation of tree construction
- [ ] Support different distance metrics (Manhattan, cosine)
- [ ] Export search paths for analysis

## 📝 License

This educational app is part of the ml-teaching repository.

## 🎉 Tips for Best Experience

1. **Start simple**: Use 100-150 points for clearer visualizations
2. **Try edge cases**: Move query to corners, between clusters
3. **Experiment**: Adjust parameters and observe changes
4. **Compare**: Look at all three methods for same query
5. **Read explanations**: Click on expanders for deeper insights

Enjoy teaching K-Nearest Neighbors! 🚀

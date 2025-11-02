"""
K-Nearest Neighbors: Exact vs Approximate Methods
Step-by-step visualization of KD-Trees and LSH for fast nearest neighbor search

Educational app showing:
1. Brute force KNN (baseline)
2. KD-Tree construction and search with edge cases
3. LSH (Locality Sensitive Hashing) with multiple hash functions
4. Time complexity comparison and trade-offs
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors
import time
from typing import List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="KNN: Exact vs Approximate", layout="wide", page_icon="🔍")

# Color scheme
COLORS = {
    'query': '#FF6B6B',      # Red for query point
    'neighbors': '#4ECDC4',   # Teal for true neighbors
    'candidate': '#95E1D3',   # Light teal for candidates
    'background': '#F7F7F7',  # Light gray background
    'kdtree_split': '#FFD93D', # Yellow for KD-tree splits
    'lsh_bucket': '#A8E6CF',  # Light green for LSH buckets
    'text': '#2C3E50',        # Dark blue-gray for text
    'highlight': '#FF8B94',   # Pink for highlights
}

# Title
st.title("🔍 K-Nearest Neighbors: Exact vs Approximate Methods")
st.markdown("""
Learn how to find nearest neighbors **fast** using KD-Trees and LSH!
Compare **exact** search (slow but correct) vs **approximate** search (fast but may miss some neighbors).
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# Dataset settings
n_points = st.sidebar.slider("Number of points", 50, 500, 150, 50)
k_neighbors = st.sidebar.slider("K (number of neighbors)", 1, 20, 5)
random_seed = st.sidebar.slider("Random seed", 0, 100, 42)

np.random.seed(random_seed)

# Generate dataset
@st.cache_data
def generate_dataset(n, seed):
    np.random.seed(seed)
    # Create clusters for interesting structure
    n_clusters = 4
    points_per_cluster = n // n_clusters

    data = []
    cluster_centers = np.random.rand(n_clusters, 2) * 0.8 + 0.1

    for center in cluster_centers:
        cluster_points = np.random.randn(points_per_cluster, 2) * 0.1 + center
        data.append(cluster_points)

    # Add some random points
    remaining = n - points_per_cluster * n_clusters
    if remaining > 0:
        data.append(np.random.rand(remaining, 2))

    data = np.vstack(data)
    # Clip to [0, 1]
    data = np.clip(data, 0, 1)
    return data

data = generate_dataset(n_points, random_seed)

# Query point
query_x = st.sidebar.slider("Query X", 0.0, 1.0, 0.5, 0.01)
query_y = st.sidebar.slider("Query Y", 0.0, 1.0, 0.5, 0.01)
query_point = np.array([[query_x, query_y]])

# ============= TAB STRUCTURE =============
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Brute Force (Exact)",
    "🌲 KD-Tree",
    "🔨 LSH (Locality Sensitive Hashing)",
    "📊 Comparison & Complexity"
])

# ============= TAB 1: BRUTE FORCE =============
with tab1:
    st.header("Brute Force K-Nearest Neighbors")
    st.markdown(f"""
    **The baseline**: Compute distance to **all {n_points} points** and pick the {k_neighbors} closest.

    **Algorithm**:
    1. For each point in dataset: compute Euclidean distance to query
    2. Sort all distances
    3. Return K smallest

    **Time Complexity**: O(N·D) where N={n_points}, D=2 (dimensions)

    **Pros**: ✅ Always correct, simple to implement
    **Cons**: ❌ Slow for large datasets, scales linearly with N
    """)

    # Compute brute force
    start_time = time.time()
    distances = np.linalg.norm(data - query_point, axis=1)
    knn_indices = np.argsort(distances)[:k_neighbors]
    knn_distances = distances[knn_indices]
    brute_force_time = (time.time() - start_time) * 1000  # milliseconds

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=50, alpha=0.6,
               edgecolors='gray', linewidth=0.5, label='Dataset')

    # Plot query
    ax.scatter(query_point[0, 0], query_point[0, 1], c=COLORS['query'],
               s=300, marker='*', edgecolors='black', linewidth=2,
               label='Query point', zorder=10)

    # Plot neighbors
    ax.scatter(data[knn_indices, 0], data[knn_indices, 1],
               c=COLORS['neighbors'], s=150, edgecolors='black', linewidth=2,
               label=f'{k_neighbors} nearest neighbors', zorder=5)

    # Draw circles and lines to neighbors
    max_dist = knn_distances[-1]
    circle = plt.Circle((query_point[0, 0], query_point[0, 1]), max_dist,
                        color=COLORS['neighbors'], fill=False,
                        linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)

    for idx, dist in zip(knn_indices, knn_distances):
        ax.plot([query_point[0, 0], data[idx, 0]],
                [query_point[0, 1], data[idx, 1]],
                c=COLORS['neighbors'], linestyle='--', linewidth=1, alpha=0.5)

        # Annotate distance
        mid_x = (query_point[0, 0] + data[idx, 0]) / 2
        mid_y = (query_point[0, 1] + data[idx, 1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.3f}', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Brute Force: Checked ALL {n_points} points ({brute_force_time:.3f} ms)',
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # Show results
    st.subheader("📋 Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Time", f"{brute_force_time:.3f} ms")
    col2.metric("Points checked", f"{n_points}/{n_points} (100%)")
    col3.metric("Accuracy", "100% (exact)")

    with st.expander("🔍 View neighbor details"):
        for i, (idx, dist) in enumerate(zip(knn_indices, knn_distances)):
            st.write(f"**{i+1}.** Point {idx}: ({data[idx, 0]:.3f}, {data[idx, 1]:.3f}) - Distance: {dist:.4f}")

# ============= TAB 2: KD-TREE =============
with tab2:
    st.header("KD-Tree: Binary Space Partitioning")
    st.markdown(f"""
    **Idea**: Organize points in a **binary tree** by recursively splitting space along alternating axes.

    **Construction**: Recursively split data by median along alternating dimensions (X, then Y, then X, ...)
    **Search**: Traverse tree, pruning branches that can't contain closer points

    **Time Complexity**:
    - Build: O(N log N)
    - Query: O(log N) average, **O(N) worst case** (high dimensions!)

    **The catch**: Different subtree **can** contain closest point! Need backtracking.
    """)

    # KD-Tree implementation
    class KDNode:
        def __init__(self, point, idx, left=None, right=None, axis=0, depth=0):
            self.point = point
            self.idx = idx
            self.left = left
            self.right = right
            self.axis = axis
            self.depth = depth

    def build_kdtree(points, indices, depth=0):
        if len(points) == 0:
            return None

        k = points.shape[1]  # dimensions
        axis = depth % k

        # Sort points along axis
        sorted_indices = np.argsort(points[:, axis])
        median_idx = len(points) // 2

        median_pt_idx = sorted_indices[median_idx]

        return KDNode(
            point=points[median_pt_idx],
            idx=indices[median_pt_idx],
            left=build_kdtree(
                points[sorted_indices[:median_idx]],
                indices[sorted_indices[:median_idx]],
                depth + 1
            ),
            right=build_kdtree(
                points[sorted_indices[median_idx + 1:]],
                indices[sorted_indices[median_idx + 1:]],
                depth + 1
            ),
            axis=axis,
            depth=depth
        )

    # Build tree
    start_time = time.time()
    indices = np.arange(len(data))
    kdtree_root = build_kdtree(data.copy(), indices)
    build_time = (time.time() - start_time) * 1000

    st.info(f"✅ KD-Tree built in {build_time:.3f} ms")

    # Search with step tracking
    class KDTreeSearch:
        def __init__(self):
            self.visited_nodes = []
            self.pruned_nodes = []
            self.best_k = []

        def search(self, node, query, k):
            if node is None:
                return

            # Visit this node
            self.visited_nodes.append(node)

            # Compute distance
            dist = np.linalg.norm(node.point - query)

            # Update best k
            if len(self.best_k) < k:
                self.best_k.append((dist, node.idx, node.point))
                self.best_k.sort(key=lambda x: x[0])
            elif dist < self.best_k[-1][0]:
                self.best_k[-1] = (dist, node.idx, node.point)
                self.best_k.sort(key=lambda x: x[0])

            # Determine which side to visit first
            axis = node.axis
            if query[axis] < node.point[axis]:
                near_node = node.left
                far_node = node.right
            else:
                near_node = node.right
                far_node = node.left

            # Search near side
            self.search(near_node, query, k)

            # Check if we need to search far side
            # Key insight: only search far side if hypersphere crosses splitting plane!
            if len(self.best_k) < k or abs(query[axis] - node.point[axis]) < self.best_k[-1][0]:
                self.search(far_node, query, k)
            else:
                # Prune this branch
                self._add_pruned(far_node)

        def _add_pruned(self, node):
            if node is not None:
                self.pruned_nodes.append(node)
                self._add_pruned(node.left)
                self._add_pruned(node.right)

    # Perform search
    start_time = time.time()
    searcher = KDTreeSearch()
    searcher.search(kdtree_root, query_point[0], k_neighbors)
    search_time = (time.time() - start_time) * 1000

    kd_knn_indices = [idx for _, idx, _ in searcher.best_k]
    kd_knn_distances = [dist for dist, _, _ in searcher.best_k]

    # Calculate accuracy
    visited_count = len(searcher.visited_nodes)
    pruned_count = len(searcher.pruned_nodes)
    accuracy = 100 * len(set(kd_knn_indices) & set(knn_indices)) / k_neighbors

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Search Visualization")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all points
        ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=50, alpha=0.4,
                   edgecolors='gray', linewidth=0.5, zorder=1)

        # Highlight visited nodes
        visited_points = np.array([node.point for node in searcher.visited_nodes])
        ax.scatter(visited_points[:, 0], visited_points[:, 1],
                   c='orange', s=100, alpha=0.6, edgecolors='black', linewidth=1,
                   label=f'Visited ({visited_count})', zorder=3)

        # Highlight pruned nodes
        if searcher.pruned_nodes:
            pruned_points = np.array([node.point for node in searcher.pruned_nodes])
            ax.scatter(pruned_points[:, 0], pruned_points[:, 1],
                       c='red', s=80, alpha=0.3, marker='x', linewidth=2,
                       label=f'Pruned ({pruned_count})', zorder=2)

        # Plot query
        ax.scatter(query_point[0, 0], query_point[0, 1], c=COLORS['query'],
                   s=300, marker='*', edgecolors='black', linewidth=2,
                   label='Query', zorder=10)

        # Plot found neighbors
        kd_neighbor_points = np.array([pt for _, _, pt in searcher.best_k])
        ax.scatter(kd_neighbor_points[:, 0], kd_neighbor_points[:, 1],
                   c=COLORS['neighbors'], s=150, edgecolors='black', linewidth=2,
                   label=f'Found neighbors ({k_neighbors})', zorder=5)

        # Draw search radius
        if len(searcher.best_k) > 0:
            max_dist = searcher.best_k[-1][0]
            circle = plt.Circle((query_point[0, 0], query_point[0, 1]), max_dist,
                                color=COLORS['neighbors'], fill=False,
                                linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'KD-Tree Search: Visited {visited_count}/{n_points} points',
                     fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🌲 Tree Structure with Splits")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw splitting lines recursively
        def draw_splits(node, bounds, depth=0):
            if node is None:
                return

            x_min, x_max, y_min, y_max = bounds
            axis = node.axis

            # Draw splitting line
            if axis == 0:  # vertical split
                ax.plot([node.point[0], node.point[0]], [y_min, y_max],
                        c=COLORS['kdtree_split'], linewidth=2, alpha=0.7-depth*0.05,
                        zorder=1)
                # Recurse
                if node.left:
                    draw_splits(node.left, (x_min, node.point[0], y_min, y_max), depth+1)
                if node.right:
                    draw_splits(node.right, (node.point[0], x_max, y_min, y_max), depth+1)
            else:  # horizontal split
                ax.plot([x_min, x_max], [node.point[1], node.point[1]],
                        c=COLORS['kdtree_split'], linewidth=2, alpha=0.7-depth*0.05,
                        zorder=1)
                # Recurse
                if node.left:
                    draw_splits(node.left, (x_min, x_max, y_min, node.point[1]), depth+1)
                if node.right:
                    draw_splits(node.right, (x_min, x_max, node.point[1], y_max), depth+1)

        draw_splits(kdtree_root, (-0.05, 1.05, -0.05, 1.05))

        # Plot points
        ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=50, alpha=0.6,
                   edgecolors='gray', linewidth=0.5, zorder=2)

        # Highlight query and neighbors
        ax.scatter(query_point[0, 0], query_point[0, 1], c=COLORS['query'],
                   s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(kd_neighbor_points[:, 0], kd_neighbor_points[:, 1],
                   c=COLORS['neighbors'], s=150, edgecolors='black', linewidth=2, zorder=5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title('Space Partitioning by KD-Tree', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    # Metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Build time", f"{build_time:.3f} ms")
    col2.metric("Search time", f"{search_time:.3f} ms")
    col3.metric("Speedup", f"{brute_force_time / search_time:.1f}×")
    col4.metric("Accuracy", f"{accuracy:.0f}%")

    # Important edge case explanation
    st.markdown("---")
    st.subheader("⚠️ Important Edge Case: Backtracking Required!")

    st.markdown("""
    **Key insight**: The closest point might be in a **different branch** of the tree!

    **Why?** The splitting plane only considers **one dimension**. A point might be:
    - Far away along the splitting dimension
    - But very close in the other dimension(s)

    **Solution**: Check if the hypersphere (search radius) **crosses** the splitting plane.
    If yes, we must search the "far" branch too!
    """)

    # Show example of pruning
    st.info(f"""
    **In this search:**
    - ✅ Visited: {visited_count} nodes (out of {n_points})
    - ✂️ Pruned: {pruned_count} nodes (branches that definitely don't contain closer points)
    - 📊 Efficiency: Checked only {100*visited_count/n_points:.1f}% of points!
    """)

    with st.expander("🎓 Why KD-Trees fail in high dimensions"):
        st.markdown("""
        **The curse of dimensionality**: KD-Trees become inefficient when D (dimensions) > 20

        **Reason**: In high dimensions:
        1. The hypersphere almost **always** crosses splitting planes
        2. Can't prune many branches → must check most points
        3. Degrades to O(N) instead of O(log N)

        **This is why we need other methods (like LSH) for high-dimensional data!**
        """)

# ============= TAB 3: LSH =============
with tab3:
    st.header("LSH: Locality Sensitive Hashing")
    st.markdown("""
    **Big idea**: Use **hash functions** that put similar points in the same bucket!

    **Unlike normal hashing** (where similar inputs → different hashes),
    LSH ensures: **similar inputs → same hash with high probability**

    **For Euclidean space**: Use random hyperplane projections
    """)

    # LSH settings
    n_hash_tables = st.slider("Number of hash tables", 1, 10, 5,
                              help="More tables = better recall but slower")
    n_hash_bits = st.slider("Hash bits per table", 2, 8, 4,
                            help="More bits = finer buckets but sparser")

    # LSH implementation
    class LSH:
        def __init__(self, n_tables, n_bits, data):
            self.n_tables = n_tables
            self.n_bits = n_bits
            self.data = data
            self.n_points = len(data)
            self.dim = data.shape[1]

            # Generate random hyperplanes for each table
            self.hyperplanes = []
            for _ in range(n_tables):
                # Each table has n_bits random hyperplanes
                planes = np.random.randn(n_bits, self.dim)
                planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
                self.hyperplanes.append(planes)

            # Build hash tables
            self.hash_tables = [dict() for _ in range(n_tables)]
            for i, point in enumerate(data):
                for t in range(n_tables):
                    hash_val = self._hash(point, t)
                    if hash_val not in self.hash_tables[t]:
                        self.hash_tables[t][hash_val] = []
                    self.hash_tables[t][hash_val].append(i)

        def _hash(self, point, table_idx):
            """Hash a point using hyperplanes from table_idx"""
            planes = self.hyperplanes[table_idx]
            # Compute dot products
            projections = np.dot(planes, point)
            # Convert to binary hash
            bits = (projections >= 0).astype(int)
            # Convert to integer
            hash_val = int(''.join(map(str, bits)), 2)
            return hash_val

        def query(self, point, k):
            """Find k nearest neighbors using LSH"""
            candidates = set()
            hash_values = []

            # Get candidates from all tables
            for t in range(self.n_tables):
                hash_val = self._hash(point, t)
                hash_values.append(hash_val)

                if hash_val in self.hash_tables[t]:
                    candidates.update(self.hash_tables[t][hash_val])

            # If not enough candidates, return what we have
            if len(candidates) == 0:
                # Fallback: return random points
                candidates = set(np.random.choice(self.n_points, min(k, self.n_points), replace=False))

            # Compute distances to candidates only
            candidate_list = list(candidates)
            candidate_points = self.data[candidate_list]
            distances = np.linalg.norm(candidate_points - point.reshape(1, -1), axis=1)

            # Get top k
            if len(candidate_list) >= k:
                top_k_idx = np.argsort(distances)[:k]
                result_indices = [candidate_list[i] for i in top_k_idx]
                result_distances = distances[top_k_idx]
            else:
                result_indices = candidate_list
                result_distances = distances

            return result_indices, result_distances, candidate_list, hash_values

    # Build LSH
    start_time = time.time()
    lsh = LSH(n_hash_tables, n_hash_bits, data)
    lsh_build_time = (time.time() - start_time) * 1000

    st.success(f"✅ LSH built with {n_hash_tables} tables × {n_hash_bits} bits in {lsh_build_time:.3f} ms")

    # Query
    start_time = time.time()
    lsh_indices, lsh_distances, candidates, hash_vals = lsh.query(query_point[0], k_neighbors)
    lsh_search_time = (time.time() - start_time) * 1000

    # Calculate accuracy
    lsh_accuracy = 100 * len(set(lsh_indices) & set(knn_indices)) / k_neighbors

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 LSH Search Process")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all points
        ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=50, alpha=0.4,
                   edgecolors='gray', linewidth=0.5, zorder=1, label='All points')

        # Highlight candidates
        if candidates:
            candidate_points = data[candidates]
            ax.scatter(candidate_points[:, 0], candidate_points[:, 1],
                       c=COLORS['candidate'], s=120, alpha=0.6, edgecolors='orange', linewidth=1.5,
                       label=f'Candidates ({len(candidates)})', zorder=3)

        # Plot query
        ax.scatter(query_point[0, 0], query_point[0, 1], c=COLORS['query'],
                   s=300, marker='*', edgecolors='black', linewidth=2,
                   label='Query', zorder=10)

        # Plot found neighbors
        if lsh_indices:
            lsh_neighbor_points = data[lsh_indices]
            ax.scatter(lsh_neighbor_points[:, 0], lsh_neighbor_points[:, 1],
                       c=COLORS['neighbors'], s=150, edgecolors='black', linewidth=2,
                       label=f'LSH neighbors ({len(lsh_indices)})', zorder=5)

        # Plot true neighbors for comparison (with different marker)
        true_neighbor_points = data[knn_indices]
        ax.scatter(true_neighbor_points[:, 0], true_neighbor_points[:, 1],
                   s=200, facecolors='none', edgecolors='red', linewidth=2.5,
                   marker='o', label='True neighbors', zorder=4)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'LSH: Checked {len(candidates)}/{n_points} candidates ({100*len(candidates)/n_points:.1f}%)',
                     fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🔨 Hash Functions (First Table)")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot points
        ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=50, alpha=0.4,
                   edgecolors='gray', linewidth=0.5, zorder=1)

        # Draw hyperplanes from first table
        for i, plane in enumerate(lsh.hyperplanes[0][:min(4, n_hash_bits)]):
            # Normal vector is plane
            # Hyperplane: plane · x = 0
            # For visualization, draw line through origin with normal = plane

            # Convert normal to line direction (perpendicular)
            direction = np.array([-plane[1], plane[0]])

            # Draw line
            t_vals = np.linspace(-2, 2, 100)
            line_x = 0.5 + direction[0] * t_vals
            line_y = 0.5 + direction[1] * t_vals

            # Clip to plot bounds
            valid = (line_x >= -0.05) & (line_x <= 1.05) & (line_y >= -0.05) & (line_y <= 1.05)

            color = plt.cm.Set3(i / max(4, n_hash_bits))
            ax.plot(line_x[valid], line_y[valid], linewidth=2.5, alpha=0.8,
                    color=color, label=f'Hyperplane {i}', zorder=2)

        # Plot query
        ax.scatter(query_point[0, 0], query_point[0, 1], c=COLORS['query'],
                   s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'Random Hyperplanes (Table 0)', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    # Metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Build time", f"{lsh_build_time:.3f} ms")
    col2.metric("Search time", f"{lsh_search_time:.3f} ms")
    col3.metric("Speedup vs brute force", f"{brute_force_time / lsh_search_time:.1f}×")
    col4.metric("Accuracy (recall)", f"{lsh_accuracy:.0f}%")

    # Show hash values
    st.subheader("🔢 Hash Values for Query Point")
    hash_cols = st.columns(min(5, n_hash_tables))
    for i, hash_val in enumerate(hash_vals):
        with hash_cols[i % len(hash_cols)]:
            binary = format(hash_val, f'0{n_hash_bits}b')
            st.metric(f"Table {i}", binary, f"Decimal: {hash_val}")

    # Explanations
    st.markdown("---")
    st.subheader("🎓 How LSH Works")

    st.markdown(f"""
    **Step 1: Hash Construction** ({n_hash_tables} tables, {n_hash_bits} bits each)
    - Generate {n_hash_bits} random hyperplanes for each table
    - Each hyperplane divides space into 2 half-spaces
    - For each point: compute which side of each hyperplane (+ or -)
    - Combine {n_hash_bits} bits → hash value (0 to {2**n_hash_bits - 1})

    **Step 2: Index Building**
    - For each point in dataset: compute hash for each table
    - Store point in corresponding bucket (hash value)

    **Step 3: Query**
    - Hash query point using same hyperplanes
    - Retrieve all points in same buckets across all tables
    - These are **candidates** (likely neighbors)
    - Compute exact distances only to candidates
    - Return top K

    **Key Parameters:**
    - **More tables** → Higher recall (find more true neighbors) but slower
    - **More bits** → Finer buckets (fewer candidates) but might miss neighbors
    """)

    with st.expander("⚖️ LSH Trade-offs"):
        st.markdown("""
        **Advantages:**
        - ✅ **Sub-linear time**: O(n^ρ) where ρ < 1
        - ✅ **Works in high dimensions** (unlike KD-trees!)
        - ✅ **Tunable**: Adjust tables/bits for speed vs accuracy

        **Disadvantages:**
        - ❌ **Approximate**: May miss true nearest neighbors
        - ❌ **Memory overhead**: Multiple hash tables
        - ❌ **Parameter tuning**: Need to choose # tables and bits

        **When to use LSH:**
        - High-dimensional data (D > 20)
        - Large datasets (N > 1M)
        - Can tolerate approximate results
        - Need fast queries
        """)

    with st.expander("🔧 Parameter Tuning Guide"):
        st.markdown(f"""
        **Current settings**: {n_hash_tables} tables × {n_hash_bits} bits

        **If accuracy too low ({lsh_accuracy:.0f}% < 80%)**:
        - ✅ Increase number of tables (more chances to find neighbors)
        - ⚠️ Decrease hash bits (larger buckets, more candidates)

        **If search too slow**:
        - ✅ Decrease number of tables
        - ✅ Increase hash bits (smaller buckets, fewer candidates)

        **Rule of thumb**:
        - Tables: 5-10 for most applications
        - Bits: log2(N/K) where N = dataset size, K = desired neighbors

        **Try it**: Adjust sliders above and see how results change!
        """)

# ============= TAB 4: COMPARISON =============
with tab4:
    st.header("📊 Method Comparison")

    st.markdown("""
    Compare all three methods side-by-side: speed, accuracy, and when to use each.
    """)

    # Summary table
    st.subheader("⚡ Performance Summary")

    import pandas as pd

    comparison_data = {
        'Method': ['Brute Force', 'KD-Tree', 'LSH'],
        'Build Time (ms)': ['-', f'{build_time:.3f}', f'{lsh_build_time:.3f}'],
        'Query Time (ms)': [f'{brute_force_time:.3f}', f'{search_time:.3f}', f'{lsh_search_time:.3f}'],
        'Speedup': ['1.0×', f'{brute_force_time/search_time:.1f}×', f'{brute_force_time/lsh_search_time:.1f}×'],
        'Points Checked': [f'{n_points}/{n_points}', f'{visited_count}/{n_points}', f'{len(candidates)}/{n_points}'],
        'Accuracy': ['100%', f'{accuracy:.0f}%', f'{lsh_accuracy:.0f}%'],
    }

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Time complexity
    st.markdown("---")
    st.subheader("⏱️ Time Complexity Analysis")

    complexity_md = f"""
    | Method | Build | Query (Average) | Query (Worst) | Space |
    |--------|-------|----------------|---------------|--------|
    | **Brute Force** | O(1) | O(N·D) | O(N·D) | O(N·D) |
    | **KD-Tree** | O(N log N) | **O(log N)** | O(N) 😱 | O(N) |
    | **LSH** | O(N·L·B) | **O(N^ρ)** where ρ<1 | O(N^ρ) | O(N·L) |

    Where:
    - N = {n_points} (number of points)
    - D = 2 (dimensions)
    - L = {n_hash_tables} (number of hash tables)
    - B = {n_hash_bits} (hash bits per table)
    - ρ ≈ 0.5-0.8 (depends on parameters and data distribution)

    **Key insights:**
    - 🟢 **KD-Tree**: Best average case, but degrades in high dimensions
    - 🔵 **LSH**: Consistent performance even in high dimensions
    - 🔴 **Brute Force**: Always O(N), but simple and always correct
    """
    st.markdown(complexity_md)

    # When to use what
    st.markdown("---")
    st.subheader("🎯 Which Method to Use?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔴 Brute Force**

        **Use when:**
        - Small dataset (N < 10K)
        - Need exact results
        - Simple implementation

        **Avoid when:**
        - Large dataset
        - Real-time queries needed
        """)

    with col2:
        st.markdown("""
        **🌲 KD-Tree**

        **Use when:**
        - Low dimensions (D < 20)
        - Need exact results
        - Medium-large dataset

        **Avoid when:**
        - High dimensions (curse of dimensionality)
        - D > 50
        """)

    with col3:
        st.markdown("""
        **🔨 LSH**

        **Use when:**
        - High dimensions (D > 20)
        - Huge dataset (N > 1M)
        - Can tolerate ~90% accuracy

        **Avoid when:**
        - Need 100% accuracy
        - Low dimensions (overkill)
        """)

    # Visual comparison
    st.markdown("---")
    st.subheader("📈 Visual Comparison: Query Time vs Dataset Size")

    # Simulate scaling
    sizes = [100, 500, 1000, 5000, 10000, 50000]

    # Approximate complexity
    brute_times = [s * 0.01 for s in sizes]  # O(N)
    kd_times = [np.log2(s) * 0.1 for s in sizes]  # O(log N)
    lsh_times = [s**0.6 * 0.005 for s in sizes]  # O(N^0.6)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=sizes, y=brute_times, mode='lines+markers',
                             name='Brute Force O(N)',
                             line=dict(color='red', width=3)))

    fig.add_trace(go.Scatter(x=sizes, y=kd_times, mode='lines+markers',
                             name='KD-Tree O(log N)',
                             line=dict(color='green', width=3)))

    fig.add_trace(go.Scatter(x=sizes, y=lsh_times, mode='lines+markers',
                             name='LSH O(N^0.6)',
                             line=dict(color='blue', width=3)))

    fig.update_layout(
        title='Query Time Scaling (Lower is Better)',
        xaxis_title='Dataset Size (N)',
        yaxis_title='Query Time (ms)',
        xaxis_type='log',
        yaxis_type='log',
        height=500,
        template='plotly_white',
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Note**: This is a theoretical comparison. Actual performance depends on:
    - Data distribution (clustered vs uniform)
    - Implementation quality
    - Hardware (CPU cache, vectorization)
    - Dimensionality (KD-Tree degrades badly in high-D)
    """)

    # Real world applications
    st.markdown("---")
    st.subheader("🌍 Real-World Applications")

    st.markdown("""
    **Image Search** (e.g., Google Images)
    - Extract deep features (ResNet, CLIP) → 512-2048 dimensions
    - Use **LSH** for initial retrieval (millions of images)
    - Re-rank top candidates with exact distance

    **Recommendation Systems**
    - User/item embeddings (50-300 dimensions)
    - Use **LSH** or **HNSW** (Hierarchical Navigable Small World)
    - Find similar users/items in milliseconds

    **Duplicate Detection**
    - Document similarity (TF-IDF, BERT embeddings)
    - Use **LSH** with MinHash for Jaccard similarity
    - Find near-duplicates in large corpora

    **Robotics & Autonomous Driving**
    - Point cloud processing (3D LIDAR data)
    - Use **KD-Tree** for 3D nearest neighbors (D=3)
    - Real-time obstacle detection and mapping

    **Database Query Optimization**
    - Spatial queries (PostGIS)
    - Use **R-Tree** (variant of KD-Tree) for geographic data
    - "Find all restaurants within 1km"
    """)

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Key Takeaways

1. **Brute Force**: Simple, exact, but slow O(N)
2. **KD-Tree**: Fast O(log N) average case, but fails in high dimensions
3. **LSH**: Sub-linear O(N^ρ), works in high-D, but approximate

**The catch with KD-Tree**: Different subtree CAN contain closest point! Must check if search radius crosses splitting plane.

**The power of LSH**: Hash functions that preserve locality → similar points hash together!

**In practice**: Often use **hybrid approaches**:
- LSH for initial retrieval (get top 1000 candidates)
- Brute force on candidates (find exact top 10)
- Best of both worlds: Fast + Accurate!

**Next level**: Explore advanced methods like HNSW, FAISS, ScaNN for production systems.
""")

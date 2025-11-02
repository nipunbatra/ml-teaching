"""
KNN Approximation: Step-by-Step Algorithm Walkthrough
Interactive visualization showing construction and querying of KD-Trees and LSH

Shows:
1. KD-Tree construction - each split visualized
2. KD-Tree query - decision at each node
3. LSH construction - hash table building
4. LSH query - hashing and bucket lookup
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from typing import List, Tuple, Optional, Dict
import time

# Page config
st.set_page_config(page_title="KNN: Step-by-Step", layout="wide", page_icon="🎬")

# Beautiful color scheme
COLORS = {
    'query': '#E63946',          # Bright red
    'neighbors': '#06FFA5',       # Bright teal
    'current': '#FFB703',         # Orange
    'visited': '#8338EC',         # Purple
    'unvisited': '#D3D3D3',       # Light gray
    'split_x': '#FB5607',         # Orange-red
    'split_y': '#3A86FF',         # Blue
    'bucket': '#06FFA5',          # Teal
    'background': '#FFFFFF',      # White
    'text': '#2B2D42',            # Dark gray
    'highlight': '#FFD60A',       # Yellow
    'pruned': '#EF476F',          # Pink
}

st.title("🎬 KNN Approximation: Step-by-Step")
st.markdown("**Watch algorithms come to life!** See each step of KD-Tree and LSH construction and querying.")

# Sidebar
st.sidebar.header("⚙️ Dataset Settings")
n_points = st.sidebar.slider("Number of points", 10, 50, 20, 5)
random_seed = st.sidebar.slider("Random seed", 0, 100, 42)

np.random.seed(random_seed)

# Generate dataset
@st.cache_data
def generate_clustered_data(n, seed):
    np.random.seed(seed)
    # Create 2-3 clusters for visual clarity
    n_clusters = 3
    points_per_cluster = n // n_clusters

    data = []
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]])

    for center in centers:
        cluster_points = np.random.randn(points_per_cluster, 2) * 0.12 + center
        data.append(cluster_points)

    # Add remaining points
    remaining = n - points_per_cluster * n_clusters
    if remaining > 0:
        data.append(np.random.rand(remaining, 2))

    data = np.vstack(data)
    data = np.clip(data, 0, 1)
    return data

data = generate_clustered_data(n_points, random_seed)

# Query point
query_x = st.sidebar.slider("Query X", 0.0, 1.0, 0.5, 0.05)
query_y = st.sidebar.slider("Query Y", 0.0, 1.0, 0.5, 0.05)
query_point = np.array([query_x, query_y])

k_neighbors = st.sidebar.slider("K (neighbors to find)", 1, 10, 3)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌲 KD-Tree Construction",
    "🔍 KD-Tree Query",
    "🔨 LSH Construction",
    "🎯 LSH Query"
])

# ==================== TAB 1: KD-TREE CONSTRUCTION ====================
with tab1:
    st.header("🌲 KD-Tree Construction: Step-by-Step")

    st.markdown("""
    **Watch the tree being built!** Each step shows:
    - Which points are being split
    - Which axis (X or Y) is used
    - The median point chosen as split
    - Resulting left and right subtrees
    """)

    # KD-Tree node class
    class KDNode:
        def __init__(self, point, idx, axis, depth, bounds, left=None, right=None):
            self.point = point
            self.idx = idx
            self.axis = axis  # 0 = X, 1 = Y
            self.depth = depth
            self.bounds = bounds  # (x_min, x_max, y_min, y_max)
            self.left = left
            self.right = right

    # Build tree with step recording
    class KDTreeBuilder:
        def __init__(self):
            self.steps = []
            self.nodes = []

        def build(self, points, indices, depth=0, bounds=(0, 1, 0, 1)):
            if len(points) == 0:
                return None

            axis = depth % 2  # Alternate X (0) and Y (1)

            # Record step BEFORE splitting
            self.steps.append({
                'depth': depth,
                'axis': axis,
                'points': points.copy(),
                'indices': indices.copy(),
                'bounds': bounds,
                'action': 'split',
                'median_idx': None,
            })

            # Sort and find median
            sorted_indices = np.argsort(points[:, axis])
            median_idx = len(points) // 2
            median_pt_idx = sorted_indices[median_idx]

            # Update step with median
            self.steps[-1]['median_idx'] = median_pt_idx
            self.steps[-1]['median_point'] = points[median_pt_idx]

            # Create node
            node = KDNode(
                point=points[median_pt_idx],
                idx=indices[median_pt_idx],
                axis=axis,
                depth=depth,
                bounds=bounds
            )
            self.nodes.append(node)

            # Split bounds for children
            x_min, x_max, y_min, y_max = bounds
            if axis == 0:  # X split
                left_bounds = (x_min, node.point[0], y_min, y_max)
                right_bounds = (node.point[0], x_max, y_min, y_max)
            else:  # Y split
                left_bounds = (x_min, x_max, y_min, node.point[1])
                right_bounds = (x_min, x_max, node.point[1], y_max)

            # Recurse
            left_points = points[sorted_indices[:median_idx]]
            left_indices = indices[sorted_indices[:median_idx]]

            right_points = points[sorted_indices[median_idx + 1:]]
            right_indices = indices[sorted_indices[median_idx + 1:]]

            node.left = self.build(left_points, left_indices, depth + 1, left_bounds)
            node.right = self.build(right_points, right_indices, depth + 1, right_bounds)

            return node

    # Build the tree
    builder = KDTreeBuilder()
    indices = np.arange(len(data))
    root = builder.build(data.copy(), indices)

    total_steps = len(builder.steps)

    st.info(f"**Tree has {total_steps} construction steps** (one for each split)")

    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        step_idx = st.slider("Construction Step", 0, total_steps - 1, 0,
                             help="Move slider to see each step of tree construction")

    # Show current step
    if step_idx < total_steps:
        step = builder.steps[step_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📍 Step {step_idx + 1}/{total_steps}")
            st.markdown(f"""
            **Depth**: {step['depth']}
            **Axis**: {'X (vertical split)' if step['axis'] == 0 else 'Y (horizontal split)'}
            **Points in this region**: {len(step['points'])}
            **Action**: Split at median
            """)

            if step['median_idx'] is not None:
                median_pt = step['median_point']
                st.success(f"**Median point**: ({median_pt[0]:.3f}, {median_pt[1]:.3f})")

        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 10))

            # Draw bounds of current region
            x_min, x_max, y_min, y_max = step['bounds']
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=3, edgecolor='black', facecolor='none',
                            linestyle='--', label='Current region')
            ax.add_patch(rect)

            # Draw all points (faded)
            ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=60, alpha=0.3,
                      edgecolors='gray', linewidth=0.5, zorder=1)

            # Highlight points in current region
            region_points = step['points']
            ax.scatter(region_points[:, 0], region_points[:, 1],
                      c=COLORS['unvisited'], s=120, alpha=0.8,
                      edgecolors='black', linewidth=1.5, zorder=3,
                      label=f'Points in region ({len(region_points)})')

            # Highlight median point
            if step['median_idx'] is not None:
                median_pt = step['median_point']
                ax.scatter(median_pt[0], median_pt[1],
                          c=COLORS['current'], s=300, marker='*',
                          edgecolors='black', linewidth=2, zorder=10,
                          label='Median (split point)')

                # Draw split line
                axis = step['axis']
                if axis == 0:  # X split (vertical line)
                    ax.axvline(x=median_pt[0], ymin=(y_min-0.05)/1.1, ymax=(y_max+0.05)/1.1,
                              color=COLORS['split_x'], linewidth=4, linestyle='-',
                              label='Split line (X)', zorder=5, alpha=0.8)

                    # Arrows showing left/right
                    mid_y = (y_min + y_max) / 2
                    ax.annotate('LEFT', xy=(median_pt[0] - 0.08, mid_y),
                               fontsize=14, weight='bold', color=COLORS['split_x'],
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
                    ax.annotate('RIGHT', xy=(median_pt[0] + 0.02, mid_y),
                               fontsize=14, weight='bold', color=COLORS['split_x'],
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
                else:  # Y split (horizontal line)
                    ax.axhline(y=median_pt[1], xmin=(x_min-0.05)/1.1, xmax=(x_max+0.05)/1.1,
                              color=COLORS['split_y'], linewidth=4, linestyle='-',
                              label='Split line (Y)', zorder=5, alpha=0.8)

                    # Arrows showing up/down
                    mid_x = (x_min + x_max) / 2
                    ax.annotate('DOWN', xy=(mid_x, median_pt[1] - 0.08),
                               fontsize=14, weight='bold', color=COLORS['split_y'],
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
                    ax.annotate('UP', xy=(mid_x, median_pt[1] + 0.02),
                               fontsize=14, weight='bold', color=COLORS['split_y'],
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
            ax.set_title(f"Step {step_idx + 1}: Depth {step['depth']}, Split on {'X' if step['axis'] == 0 else 'Y'} axis",
                        fontsize=14, weight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X', fontsize=12, weight='bold')
            ax.set_ylabel('Y', fontsize=12, weight='bold')

            st.pyplot(fig)
            plt.close()

    # Show final tree structure
    st.markdown("---")
    st.subheader("🌳 Complete Tree Structure")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw all splits
    def draw_all_splits(node):
        if node is None:
            return

        x_min, x_max, y_min, y_max = node.bounds

        # Color based on depth
        alpha = max(0.3, 1.0 - node.depth * 0.15)

        if node.axis == 0:  # X split
            ax.plot([node.point[0], node.point[0]], [y_min, y_max],
                   color=COLORS['split_x'], linewidth=2.5, alpha=alpha, zorder=2)
        else:  # Y split
            ax.plot([x_min, x_max], [node.point[1], node.point[1]],
                   color=COLORS['split_y'], linewidth=2.5, alpha=alpha, zorder=2)

        draw_all_splits(node.left)
        draw_all_splits(node.right)

    draw_all_splits(root)

    # Draw points
    ax.scatter(data[:, 0], data[:, 1], c='white', s=150,
              edgecolors='black', linewidth=2, zorder=5)

    # Label points
    for i, (x, y) in enumerate(data):
        ax.text(x, y, str(i), fontsize=9, weight='bold',
               ha='center', va='center', zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title('Complete KD-Tree: All Splits', fontsize=16, weight='bold', pad=15)
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['split_x'], lw=3, label='X-axis splits (vertical)'),
        Line2D([0], [0], color=COLORS['split_y'], lw=3, label='Y-axis splits (horizontal)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    st.pyplot(fig)
    plt.close()

# ==================== TAB 2: KD-TREE QUERY ====================
with tab2:
    st.header("🔍 KD-Tree Query: Step-by-Step")

    st.markdown("""
    **Watch the search!** Each step shows:
    - Current node being examined
    - Distance calculation
    - Decision: go left or right?
    - Backtracking when needed
    - Final K nearest neighbors
    """)

    # Query algorithm with step recording
    class KDTreeQuery:
        def __init__(self, root, query, k):
            self.root = root
            self.query = query
            self.k = k
            self.steps = []
            self.best_neighbors = []  # List of (distance, node)

        def search(self, node=None, first_call=True):
            if first_call:
                node = self.root
                self.best_neighbors = []
                self.steps = []

            if node is None:
                return

            # STEP 1: Visit node
            dist = np.linalg.norm(node.point - self.query)

            self.steps.append({
                'action': 'visit',
                'node': node,
                'distance': dist,
                'best_neighbors': [n for n in self.best_neighbors],  # Copy
                'decision': None
            })

            # Update best neighbors
            if len(self.best_neighbors) < self.k:
                self.best_neighbors.append((dist, node))
                self.best_neighbors.sort(key=lambda x: x[0])
            elif dist < self.best_neighbors[-1][0]:
                self.best_neighbors[-1] = (dist, node)
                self.best_neighbors.sort(key=lambda x: x[0])

            # STEP 2: Decide which branch to visit first
            axis = node.axis
            if self.query[axis] < node.point[axis]:
                near_node = node.left
                far_node = node.right
                direction = 'left'
            else:
                near_node = node.right
                far_node = node.left
                direction = 'right'

            self.steps[-1]['decision'] = f'Go {direction} first (query.{["x","y"][axis]}={self.query[axis]:.3f} {"<" if direction=="left" else ">"} node.{["x","y"][axis]}={node.point[axis]:.3f})'

            # STEP 3: Search near branch
            if near_node is not None:
                self.steps.append({
                    'action': 'descend',
                    'node': node,
                    'direction': direction,
                    'best_neighbors': [n for n in self.best_neighbors],
                })
            self.search(near_node, False)

            # STEP 4: Check if need to search far branch
            need_backtrack = (len(self.best_neighbors) < self.k or
                            abs(self.query[axis] - node.point[axis]) < self.best_neighbors[-1][0])

            if need_backtrack and far_node is not None:
                far_direction = 'right' if direction == 'left' else 'left'
                self.steps.append({
                    'action': 'backtrack',
                    'node': node,
                    'direction': far_direction,
                    'reason': f'Search radius ({self.best_neighbors[-1][0]:.3f}) crosses split plane!',
                    'best_neighbors': [n for n in self.best_neighbors],
                })
                self.search(far_node, False)
            elif far_node is not None:
                self.steps.append({
                    'action': 'prune',
                    'node': node,
                    'direction': 'right' if direction == 'left' else 'left',
                    'reason': f'Search radius ({self.best_neighbors[-1][0]:.3f}) does not cross split plane',
                    'best_neighbors': [n for n in self.best_neighbors],
                })

    # Perform query
    searcher = KDTreeQuery(root, query_point, k_neighbors)
    searcher.search()

    total_query_steps = len(searcher.steps)

    st.info(f"**Query has {total_query_steps} steps** (visits, decisions, backtracks)")

    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query_step_idx = st.slider("Query Step", 0, total_query_steps - 1, 0,
                                   help="Move slider to see each step of the search")

    # Show current step
    if query_step_idx < total_query_steps:
        step = searcher.steps[query_step_idx]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"📍 Step {query_step_idx + 1}/{total_query_steps}")

            action = step['action']
            if action == 'visit':
                st.markdown(f"""
                **Action**: 🔍 Visit node
                **Point**: ({step['node'].point[0]:.3f}, {step['node'].point[1]:.3f})
                **Distance**: {step['distance']:.4f}
                **Current best K**: {len(step['best_neighbors'])}/{k_neighbors}
                """)

                if step['decision']:
                    st.info(f"**Decision**: {step['decision']}")

            elif action == 'descend':
                st.markdown(f"""
                **Action**: ⬇️ Descend {step['direction']}
                **From node**: ({step['node'].point[0]:.3f}, {step['node'].point[1]:.3f})
                """)

            elif action == 'backtrack':
                st.markdown(f"""
                **Action**: ↩️ Backtrack to check {step['direction']} branch
                **From node**: ({step['node'].point[0]:.3f}, {step['node'].point[1]:.3f})
                **Reason**: {step['reason']}
                """)
                st.warning("⚠️ **This is the key insight!** Must check other branch too.")

            elif action == 'prune':
                st.markdown(f"""
                **Action**: ✂️ Prune {step['direction']} branch
                **From node**: ({step['node'].point[0]:.3f}, {step['node'].point[1]:.3f})
                **Reason**: {step['reason']}
                """)
                st.success("✅ **Optimization!** Can skip this entire subtree.")

            # Show current best neighbors
            if step['best_neighbors']:
                st.markdown("**Current best neighbors:**")
                for i, (dist, node) in enumerate(step['best_neighbors']):
                    st.write(f"{i+1}. Distance: {dist:.4f}")

        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 10))

            # Draw all splits (faded)
            def draw_all_splits_faded(node, alpha=0.15):
                if node is None:
                    return
                x_min, x_max, y_min, y_max = node.bounds
                if node.axis == 0:
                    ax.plot([node.point[0], node.point[0]], [y_min, y_max],
                           color='gray', linewidth=1.5, alpha=alpha, zorder=1)
                else:
                    ax.plot([x_min, x_max], [node.point[1], node.point[1]],
                           color='gray', linewidth=1.5, alpha=alpha, zorder=1)
                draw_all_splits_faded(node.left, alpha)
                draw_all_splits_faded(node.right, alpha)

            draw_all_splits_faded(root)

            # Draw all points (faded)
            ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.4,
                      edgecolors='gray', linewidth=1, zorder=2)

            # Draw query point
            ax.scatter(query_point[0], query_point[1], c=COLORS['query'], s=400,
                      marker='*', edgecolors='black', linewidth=2.5, zorder=10,
                      label='Query')

            # Draw current node
            current_node = step['node']
            ax.scatter(current_node.point[0], current_node.point[1],
                      c=COLORS['current'], s=300, edgecolors='black', linewidth=3,
                      zorder=9, label='Current node')

            # Draw line from query to current
            ax.plot([query_point[0], current_node.point[0]],
                   [query_point[1], current_node.point[1]],
                   'k--', linewidth=2, alpha=0.5, zorder=3)

            if 'distance' in step:
                mid_x = (query_point[0] + current_node.point[0]) / 2
                mid_y = (query_point[1] + current_node.point[1]) / 2
                ax.text(mid_x, mid_y, f'd={step["distance"]:.3f}',
                       fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9))

            # Draw current best neighbors
            if step['best_neighbors']:
                for dist, node in step['best_neighbors']:
                    ax.scatter(node.point[0], node.point[1],
                              c=COLORS['neighbors'], s=200, edgecolors='black',
                              linewidth=2, zorder=8, alpha=0.7)

                # Draw search radius
                max_dist = step['best_neighbors'][-1][0]
                circle = Circle(query_point, max_dist, fill=False,
                              edgecolor=COLORS['neighbors'], linewidth=2.5,
                              linestyle='--', zorder=4, label=f'Search radius ({max_dist:.3f})')
                ax.add_patch(circle)

            # Highlight action-specific elements
            if action in ['descend', 'backtrack', 'prune']:
                # Highlight the split line
                node = step['node']
                x_min, x_max, y_min, y_max = node.bounds
                if node.axis == 0:
                    color = COLORS['split_x'] if action != 'prune' else COLORS['pruned']
                    ax.plot([node.point[0], node.point[0]], [y_min, y_max],
                           color=color, linewidth=4, alpha=0.8, zorder=5)
                else:
                    color = COLORS['split_y'] if action != 'prune' else COLORS['pruned']
                    ax.plot([x_min, x_max], [node.point[1], node.point[1]],
                           color=color, linewidth=4, alpha=0.8, zorder=5)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

            title = f"Step {query_step_idx + 1}: "
            if action == 'visit':
                title += "Visit Node"
            elif action == 'descend':
                title += f"Descend {step['direction'].upper()}"
            elif action == 'backtrack':
                title += f"Backtrack to {step['direction'].upper()}"
            elif action == 'prune':
                title += f"Prune {step['direction'].upper()} Branch"

            ax.set_title(title, fontsize=14, weight='bold', pad=15)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

    # Final result
    st.markdown("---")
    st.subheader("🎯 Final Result")

    final_neighbors = searcher.best_neighbors

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Found {len(final_neighbors)} nearest neighbors:**")
        for i, (dist, node) in enumerate(final_neighbors):
            st.write(f"{i+1}. Point ({node.point[0]:.3f}, {node.point[1]:.3f}) - Distance: {dist:.4f}")

    with col2:
        # Final visualization
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.5,
                  edgecolors='gray', linewidth=1, zorder=2)

        ax.scatter(query_point[0], query_point[1], c=COLORS['query'], s=400,
                  marker='*', edgecolors='black', linewidth=2.5, zorder=10)

        for dist, node in final_neighbors:
            ax.scatter(node.point[0], node.point[1], c=COLORS['neighbors'],
                      s=250, edgecolors='black', linewidth=2, zorder=8)
            ax.plot([query_point[0], node.point[0]],
                   [query_point[1], node.point[1]],
                   c=COLORS['neighbors'], linewidth=2, alpha=0.6, zorder=3)

        if final_neighbors:
            max_dist = final_neighbors[-1][0]
            circle = Circle(query_point, max_dist, fill=False,
                          edgecolor=COLORS['neighbors'], linewidth=2.5,
                          linestyle='--', zorder=4)
            ax.add_patch(circle)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'{k_neighbors} Nearest Neighbors Found!', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

# ==================== TAB 3: LSH CONSTRUCTION ====================
with tab3:
    st.header("🔨 LSH Construction: Step-by-Step")

    st.markdown("""
    **Watch hash tables being built!** Each step shows:
    - Random hyperplane generation
    - Hash computation for each point
    - Points being added to buckets
    - Multiple hash tables
    """)

    # LSH settings
    n_tables = st.slider("Number of hash tables", 1, 4, 2)
    n_bits = st.slider("Hash bits per table", 2, 4, 3)

    # LSH construction with steps
    class LSHBuilder:
        def __init__(self, n_tables, n_bits, data):
            self.n_tables = n_tables
            self.n_bits = n_bits
            self.data = data
            self.steps = []
            self.hyperplanes = []
            self.hash_tables = []

        def build(self):
            # Step 1: Generate hyperplanes for each table
            for table_idx in range(self.n_tables):
                planes = []
                for bit_idx in range(self.n_bits):
                    # Random hyperplane
                    plane = np.random.randn(2)
                    plane = plane / np.linalg.norm(plane)
                    planes.append(plane)

                    self.steps.append({
                        'action': 'generate_plane',
                        'table_idx': table_idx,
                        'bit_idx': bit_idx,
                        'plane': plane.copy()
                    })

                self.hyperplanes.append(np.array(planes))
                self.hash_tables.append({})

            # Step 2: Hash each point and add to buckets
            for point_idx, point in enumerate(self.data):
                for table_idx in range(self.n_tables):
                    # Compute hash
                    planes = self.hyperplanes[table_idx]
                    projections = np.dot(planes, point)
                    bits = (projections >= 0).astype(int)
                    hash_val = int(''.join(map(str, bits)), 2)

                    # Add to bucket
                    if hash_val not in self.hash_tables[table_idx]:
                        self.hash_tables[table_idx][hash_val] = []
                    self.hash_tables[table_idx][hash_val].append(point_idx)

                    self.steps.append({
                        'action': 'hash_point',
                        'point_idx': point_idx,
                        'point': point.copy(),
                        'table_idx': table_idx,
                        'bits': bits.copy(),
                        'hash_val': hash_val,
                        'bucket_size': len(self.hash_tables[table_idx][hash_val])
                    })

    # Build LSH
    np.random.seed(random_seed + 1)  # Different seed for LSH
    builder_lsh = LSHBuilder(n_tables, n_bits, data)
    builder_lsh.build()

    total_lsh_steps = len(builder_lsh.steps)

    # Count hyperplane and hashing steps
    plane_steps = sum(1 for s in builder_lsh.steps if s['action'] == 'generate_plane')
    hash_steps = sum(1 for s in builder_lsh.steps if s['action'] == 'hash_point')

    st.info(f"**Construction has {total_lsh_steps} steps**: {plane_steps} hyperplanes generated + {hash_steps} point hashings")

    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        lsh_step_idx = st.slider("LSH Construction Step", 0, total_lsh_steps - 1, 0)

    # Show current step
    if lsh_step_idx < total_lsh_steps:
        step = builder_lsh.steps[lsh_step_idx]

        if step['action'] == 'generate_plane':
            st.subheader(f"📍 Step {lsh_step_idx + 1}/{total_lsh_steps}: Generate Hyperplane")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Table**: {step['table_idx']}
                **Bit**: {step['bit_idx']} (of {n_bits})
                **Hyperplane normal**: ({step['plane'][0]:.3f}, {step['plane'][1]:.3f})
                """)

                st.info("Hyperplane divides space into two half-spaces: + and -")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))

                # Draw points
                ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100,
                          edgecolors='black', linewidth=1, zorder=5)

                # Draw hyperplane
                plane = step['plane']
                # Line perpendicular to normal, passing through (0.5, 0.5)
                direction = np.array([-plane[1], plane[0]])
                t_vals = np.linspace(-1, 1, 100)
                line_x = 0.5 + direction[0] * t_vals
                line_y = 0.5 + direction[1] * t_vals

                valid = (line_x >= -0.05) & (line_x <= 1.05) & (line_y >= -0.05) & (line_y <= 1.05)

                color = plt.cm.Set3(step['bit_idx'])
                ax.plot(line_x[valid], line_y[valid], color=color, linewidth=4,
                       label=f"Hyperplane {step['bit_idx']}", zorder=3)

                # Draw normal vector
                ax.arrow(0.5, 0.5, plane[0] * 0.15, plane[1] * 0.15,
                        head_width=0.03, head_length=0.02, fc=color, ec='black',
                        linewidth=2, zorder=10)
                ax.text(0.5 + plane[0] * 0.2, 0.5 + plane[1] * 0.2, 'normal',
                       fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=11)
                ax.set_title(f'Table {step["table_idx"]}, Hyperplane {step["bit_idx"]}',
                           fontsize=14, weight='bold')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

        elif step['action'] == 'hash_point':
            st.subheader(f"📍 Step {lsh_step_idx + 1}/{total_lsh_steps}: Hash Point")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Point**: {step['point_idx']} at ({step['point'][0]:.3f}, {step['point'][1]:.3f})
                **Table**: {step['table_idx']}
                **Hash bits**: {' '.join(map(str, step['bits']))} (binary)
                **Hash value**: {step['hash_val']} (decimal)
                **Bucket size**: {step['bucket_size']} points
                """)

                # Show which side of each hyperplane
                st.markdown("**Hyperplane sides:**")
                for bit_idx, bit in enumerate(step['bits']):
                    st.write(f"Plane {bit_idx}: {'+ (above/right)' if bit else '- (below/left)'}")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))

                # Draw all points (faded)
                ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.4,
                          edgecolors='gray', linewidth=1, zorder=2)

                # Draw current point
                point = step['point']
                ax.scatter(point[0], point[1], c=COLORS['current'], s=300,
                          marker='*', edgecolors='black', linewidth=2.5, zorder=10,
                          label=f'Point {step["point_idx"]}')

                # Draw all hyperplanes for this table
                planes = builder_lsh.hyperplanes[step['table_idx']]
                for bit_idx, plane in enumerate(planes):
                    direction = np.array([-plane[1], plane[0]])
                    t_vals = np.linspace(-1, 1, 100)
                    line_x = 0.5 + direction[0] * t_vals
                    line_y = 0.5 + direction[1] * t_vals

                    valid = (line_x >= -0.05) & (line_x <= 1.05) & (line_y >= -0.05) & (line_y <= 1.05)

                    color = plt.cm.Set3(bit_idx)
                    ax.plot(line_x[valid], line_y[valid], color=color, linewidth=3,
                           alpha=0.7, label=f"Plane {bit_idx}: bit={step['bits'][bit_idx]}",
                           zorder=3)

                # Show hash value
                ax.text(point[0] + 0.05, point[1] + 0.05,
                       f"Hash: {step['hash_val']}",
                       fontsize=12, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
                ax.set_title(f'Hashing Point {step["point_idx"]} (Table {step["table_idx"]})',
                           fontsize=14, weight='bold')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

    # Show final hash tables
    st.markdown("---")
    st.subheader("🗄️ Final Hash Tables")

    for table_idx in range(n_tables):
        with st.expander(f"Table {table_idx} ({len(builder_lsh.hash_tables[table_idx])} buckets)"):
            hash_table = builder_lsh.hash_tables[table_idx]

            for hash_val, point_indices in sorted(hash_table.items()):
                binary = format(hash_val, f'0{n_bits}b')
                st.write(f"**Bucket {binary} ({hash_val})**: Points {point_indices}")

# ==================== TAB 4: LSH QUERY ====================
with tab4:
    st.header("🎯 LSH Query: Step-by-Step")

    st.markdown("""
    **Watch the query!** Each step shows:
    - Hashing the query point
    - Looking up buckets in each table
    - Collecting candidates
    - Computing exact distances
    - Returning top K
    """)

    # LSH query with steps
    class LSHQuery:
        def __init__(self, builder, query, k):
            self.builder = builder
            self.query = query
            self.k = k
            self.steps = []
            self.candidates = set()

        def search(self):
            # Step 1: Hash query for each table
            for table_idx in range(self.builder.n_tables):
                planes = self.builder.hyperplanes[table_idx]
                projections = np.dot(planes, self.query)
                bits = (projections >= 0).astype(int)
                hash_val = int(''.join(map(str, bits)), 2)

                self.steps.append({
                    'action': 'hash_query',
                    'table_idx': table_idx,
                    'bits': bits.copy(),
                    'hash_val': hash_val
                })

                # Step 2: Lookup bucket
                if hash_val in self.builder.hash_tables[table_idx]:
                    bucket_points = self.builder.hash_tables[table_idx][hash_val]
                    new_candidates = set(bucket_points) - self.candidates
                    self.candidates.update(bucket_points)

                    self.steps.append({
                        'action': 'lookup_bucket',
                        'table_idx': table_idx,
                        'hash_val': hash_val,
                        'bucket_points': bucket_points.copy(),
                        'new_candidates': list(new_candidates),
                        'total_candidates': len(self.candidates)
                    })
                else:
                    self.steps.append({
                        'action': 'lookup_bucket',
                        'table_idx': table_idx,
                        'hash_val': hash_val,
                        'bucket_points': [],
                        'new_candidates': [],
                        'total_candidates': len(self.candidates)
                    })

            # Step 3: Compute distances to candidates
            if len(self.candidates) > 0:
                candidate_list = list(self.candidates)
                candidate_points = self.builder.data[candidate_list]
                distances = np.linalg.norm(candidate_points - self.query.reshape(1, -1), axis=1)

                # Get top k
                if len(candidate_list) >= self.k:
                    top_k_idx = np.argsort(distances)[:self.k]
                    self.result_indices = [candidate_list[i] for i in top_k_idx]
                    self.result_distances = distances[top_k_idx]
                else:
                    self.result_indices = candidate_list
                    self.result_distances = distances

                self.steps.append({
                    'action': 'compute_distances',
                    'num_candidates': len(candidate_list),
                    'result_indices': self.result_indices.copy(),
                    'result_distances': self.result_distances.copy()
                })
            else:
                self.result_indices = []
                self.result_distances = []

    # Perform query
    query_lsh = LSHQuery(builder_lsh, query_point, k_neighbors)
    query_lsh.search()

    total_query_lsh_steps = len(query_lsh.steps)

    st.info(f"**Query has {total_query_lsh_steps} steps**")

    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        lsh_query_step_idx = st.slider("LSH Query Step", 0, total_query_lsh_steps - 1, 0)

    # Show current step
    if lsh_query_step_idx < total_query_lsh_steps:
        step = query_lsh.steps[lsh_query_step_idx]

        if step['action'] == 'hash_query':
            st.subheader(f"📍 Step {lsh_query_step_idx + 1}/{total_query_lsh_steps}: Hash Query Point")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Table**: {step['table_idx']}
                **Hash bits**: {' '.join(map(str, step['bits']))}
                **Hash value**: {step['hash_val']}
                """)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))

                ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.4,
                          edgecolors='gray', linewidth=1, zorder=2)

                ax.scatter(query_point[0], query_point[1], c=COLORS['query'], s=400,
                          marker='*', edgecolors='black', linewidth=2.5, zorder=10,
                          label='Query')

                # Draw hyperplanes
                planes = builder_lsh.hyperplanes[step['table_idx']]
                for bit_idx, plane in enumerate(planes):
                    direction = np.array([-plane[1], plane[0]])
                    t_vals = np.linspace(-1, 1, 100)
                    line_x = 0.5 + direction[0] * t_vals
                    line_y = 0.5 + direction[1] * t_vals

                    valid = (line_x >= -0.05) & (line_x <= 1.05) & (line_y >= -0.05) & (line_y <= 1.05)

                    color = plt.cm.Set3(bit_idx)
                    ax.plot(line_x[valid], line_y[valid], color=color, linewidth=3,
                           alpha=0.7, label=f"Plane {bit_idx}: bit={step['bits'][bit_idx]}",
                           zorder=3)

                ax.text(query_point[0] + 0.05, query_point[1] + 0.05,
                       f"Hash: {step['hash_val']}",
                       fontsize=12, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=10)
                ax.set_title(f'Query Hash (Table {step["table_idx"]})', fontsize=14, weight='bold')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

        elif step['action'] == 'lookup_bucket':
            st.subheader(f"📍 Step {lsh_query_step_idx + 1}/{total_query_lsh_steps}: Lookup Bucket")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Table**: {step['table_idx']}
                **Hash value**: {step['hash_val']}
                **Points in bucket**: {len(step['bucket_points'])}
                **New candidates**: {len(step['new_candidates'])}
                **Total candidates so far**: {step['total_candidates']}
                """)

                if step['bucket_points']:
                    st.success(f"✅ Found {len(step['bucket_points'])} points in this bucket!")
                else:
                    st.warning("⚠️ Empty bucket!")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))

                ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.4,
                          edgecolors='gray', linewidth=1, zorder=2)

                ax.scatter(query_point[0], query_point[1], c=COLORS['query'], s=400,
                          marker='*', edgecolors='black', linewidth=2.5, zorder=10,
                          label='Query')

                # Highlight bucket points
                if step['bucket_points']:
                    bucket_pts = data[step['bucket_points']]
                    ax.scatter(bucket_pts[:, 0], bucket_pts[:, 1],
                              c=COLORS['bucket'], s=250, edgecolors='black', linewidth=2,
                              zorder=8, label=f'Bucket points ({len(step["bucket_points"])})')

                # Highlight new candidates
                if step['new_candidates']:
                    new_cand_pts = data[step['new_candidates']]
                    ax.scatter(new_cand_pts[:, 0], new_cand_pts[:, 1],
                              c=COLORS['highlight'], s=300, marker='s',
                              edgecolors='black', linewidth=2.5,
                              zorder=9, label=f'New candidates ({len(step["new_candidates"])})')

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=11)
                ax.set_title(f'Bucket Lookup (Table {step["table_idx"]})', fontsize=14, weight='bold')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

        elif step['action'] == 'compute_distances':
            st.subheader(f"📍 Step {lsh_query_step_idx + 1}/{total_query_lsh_steps}: Compute Distances & Select Top K")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Total candidates**: {step['num_candidates']}
                **K**: {k_neighbors}
                **Results**: {len(step['result_indices'])} neighbors found
                """)

                st.markdown("**Top K neighbors:**")
                for i, (idx, dist) in enumerate(zip(step['result_indices'], step['result_distances'])):
                    st.write(f"{i+1}. Point {idx}: distance {dist:.4f}")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))

                ax.scatter(data[:, 0], data[:, 1], c='lightgray', s=100, alpha=0.4,
                          edgecolors='gray', linewidth=1, zorder=2)

                ax.scatter(query_point[0], query_point[1], c=COLORS['query'], s=400,
                          marker='*', edgecolors='black', linewidth=2.5, zorder=10,
                          label='Query')

                # Show all candidates (faded)
                if query_lsh.candidates:
                    cand_pts = data[list(query_lsh.candidates)]
                    ax.scatter(cand_pts[:, 0], cand_pts[:, 1],
                              c=COLORS['candidate'], s=150, alpha=0.5,
                              edgecolors='black', linewidth=1, zorder=5,
                              label=f'All candidates ({len(query_lsh.candidates)})')

                # Highlight top K
                if step['result_indices']:
                    result_pts = data[step['result_indices']]
                    ax.scatter(result_pts[:, 0], result_pts[:, 1],
                              c=COLORS['neighbors'], s=250, edgecolors='black',
                              linewidth=2.5, zorder=8,
                              label=f'Top {k_neighbors} neighbors')

                    for idx, dist in zip(step['result_indices'], step['result_distances']):
                        ax.plot([query_point[0], data[idx, 0]],
                               [query_point[1], data[idx, 1]],
                               c=COLORS['neighbors'], linewidth=2, alpha=0.6, zorder=3)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=11)
                ax.set_title('Final Result: Top K Neighbors', fontsize=14, weight='bold')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Summary

**KD-Tree**: Binary space partitioning with alternating axis splits
- Construction: O(N log N)
- Query: O(log N) average, O(N) worst case
- **Key insight**: Must backtrack when search radius crosses splitting plane!

**LSH**: Hash similar points to same buckets
- Construction: O(N·L·B) where L=tables, B=bits
- Query: O(N^ρ) where ρ < 1
- **Key insight**: More tables = better recall but slower!

**Use the sliders to step through and understand each algorithm in detail!** 🚀
""")

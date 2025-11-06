"""
KD-Tree Backtracking & Pruning Visualization
Interactive demo showing how KD-trees efficiently search with pruning
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# Page config
st.set_page_config(page_title="KD-Tree Search Visualization", layout="wide")

st.title("KD-Tree: Backtracking & Pruning Visualization")

st.markdown("""
This interactive visualization shows how **KD-trees** efficiently search for nearest neighbors through:
- **Spatial partitioning**: Recursive splitting of space
- **Backtracking**: Exploring promising branches
- **Pruning**: Skipping branches that can't contain better neighbors

**Click on the plot** to set a query point and watch the search algorithm in action!
""")

# KD-Tree Node
@dataclass
class KDNode:
    point: np.ndarray
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None
    axis: int = 0
    depth: int = 0
    bounds: Tuple[float, float, float, float] = None  # (xmin, xmax, ymin, ymax)

# Search step tracking
@dataclass
class SearchStep:
    node: KDNode
    action: str  # 'visit', 'consider', 'prune', 'backtrack', 'best_update'
    distance: float
    best_dist: float
    best_point: np.ndarray
    bounds_checked: bool = False

class KDTree:
    def __init__(self, points):
        self.points = np.array(points)
        self.root = None
        self.search_steps = []

    def build(self, points=None, depth=0, bounds=None):
        """Build KD-tree recursively"""
        if points is None:
            points = self.points
            bounds = (points[:, 0].min(), points[:, 0].max(),
                     points[:, 1].min(), points[:, 1].max())

        if len(points) == 0:
            return None

        axis = depth % 2  # 0 for x, 1 for y
        sorted_indices = np.argsort(points[:, axis])
        median_idx = len(points) // 2
        median_point = points[sorted_indices[median_idx]]

        # Create node
        node = KDNode(
            point=median_point,
            axis=axis,
            depth=depth,
            bounds=bounds
        )

        # Split bounds for children
        xmin, xmax, ymin, ymax = bounds
        if axis == 0:  # Split on x
            left_bounds = (xmin, median_point[0], ymin, ymax)
            right_bounds = (median_point[0], xmax, ymin, ymax)
        else:  # Split on y
            left_bounds = (xmin, xmax, ymin, median_point[1])
            right_bounds = (xmin, xmax, median_point[1], ymax)

        # Recursively build subtrees
        left_points = points[sorted_indices[:median_idx]]
        right_points = points[sorted_indices[median_idx + 1:]]

        node.left = self.build(left_points, depth + 1, left_bounds)
        node.right = self.build(right_points, depth + 1, right_bounds)

        if depth == 0:
            self.root = node
        return node

    def distance(self, p1, p2):
        """Euclidean distance"""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def bounds_distance(self, query, bounds):
        """Minimum distance from query to bounding box"""
        xmin, xmax, ymin, ymax = bounds
        dx = max(xmin - query[0], 0, query[0] - xmax)
        dy = max(ymin - query[1], 0, query[1] - ymax)
        return np.sqrt(dx**2 + dy**2)

    def search_knn(self, query, k=1):
        """Search for k nearest neighbors with step tracking"""
        self.search_steps = []
        best = []
        self._search_recursive(self.root, query, k, best)
        return best

    def _search_recursive(self, node, query, k, best):
        """Recursive KNN search with backtracking and pruning"""
        if node is None:
            return

        # Visit current node
        dist = self.distance(query, node.point)
        self.search_steps.append(SearchStep(
            node=node,
            action='visit',
            distance=dist,
            best_dist=best[0][0] if best else float('inf'),
            best_point=best[0][1] if best else None
        ))

        # Consider this point
        best.append((dist, node.point))
        best.sort(key=lambda x: x[0])
        if len(best) > k:
            best.pop()

        self.search_steps.append(SearchStep(
            node=node,
            action='consider',
            distance=dist,
            best_dist=best[0][0],
            best_point=best[0][1]
        ))

        # Determine which child to visit first
        if query[node.axis] < node.point[node.axis]:
            first_child = node.left
            second_child = node.right
        else:
            first_child = node.right
            second_child = node.left

        # Visit closer child first
        self._search_recursive(first_child, query, k, best)

        # Check if we need to visit the other child (backtracking)
        if second_child is not None:
            # Calculate distance to splitting hyperplane
            split_dist = abs(query[node.axis] - node.point[node.axis])
            bounds_dist = self.bounds_distance(query, second_child.bounds)

            if len(best) < k or bounds_dist < best[-1][0]:
                # Backtrack: other side might have closer points
                self.search_steps.append(SearchStep(
                    node=second_child,
                    action='backtrack',
                    distance=bounds_dist,
                    best_dist=best[0][0],
                    best_point=best[0][1],
                    bounds_checked=True
                ))
                self._search_recursive(second_child, query, k, best)
            else:
                # Prune: other side can't have closer points
                self.search_steps.append(SearchStep(
                    node=second_child,
                    action='prune',
                    distance=bounds_dist,
                    best_dist=best[0][0],
                    best_point=best[0][1],
                    bounds_checked=True
                ))

# Sidebar controls
st.sidebar.header("Settings")

n_points = st.sidebar.slider("Number of points", 10, 50, 20)
k_neighbors = st.sidebar.slider("K (neighbors to find)", 1, 5, 1)
random_seed = st.sidebar.slider("Random seed", 0, 100, 42)

np.random.seed(random_seed)
points = np.random.rand(n_points, 2) * 10

# Build KD-tree
kdtree = KDTree(points)
kdtree.build()

# Session state for query point
if 'query_point' not in st.session_state:
    st.session_state.query_point = np.array([5.0, 5.0])

if 'animation_step' not in st.session_state:
    st.session_state.animation_step = 0

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Visualization")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw KD-tree partitions
    def draw_partitions(node, depth=0):
        if node is None:
            return

        xmin, xmax, ymin, ymax = node.bounds

        # Draw splitting line
        if node.axis == 0:  # Vertical split
            color = 'blue' if depth % 2 == 0 else 'red'
            ax.plot([node.point[0], node.point[0]], [ymin, ymax],
                   color=color, linestyle='--', alpha=0.3, linewidth=1)
        else:  # Horizontal split
            color = 'red' if depth % 2 == 1 else 'blue'
            ax.plot([xmin, xmax], [node.point[1], node.point[1]],
                   color=color, linestyle='--', alpha=0.3, linewidth=1)

        draw_partitions(node.left, depth + 1)
        draw_partitions(node.right, depth + 1)

    draw_partitions(kdtree.root)

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=100,
              edgecolors='black', linewidth=1, zorder=2, label='Data points')

    # Run search if we have a query point
    query = st.session_state.query_point
    kdtree.search_knn(query, k_neighbors)

    # Animation controls
    show_animation = st.sidebar.checkbox("Show step-by-step animation", value=True)

    if show_animation and len(kdtree.search_steps) > 0:
        step_idx = st.sidebar.slider("Animation step", 0, len(kdtree.search_steps) - 1,
                                     st.session_state.animation_step)
        st.session_state.animation_step = step_idx

        # Show steps up to current
        visited_nodes = set()
        pruned_nodes = set()
        considered_points = []

        for i, step in enumerate(kdtree.search_steps[:step_idx + 1]):
            if step.action == 'visit':
                visited_nodes.add(tuple(step.node.point))
            elif step.action == 'prune':
                pruned_nodes.add(tuple(step.node.point))
            elif step.action == 'consider':
                considered_points.append(step.node.point)

        # Draw visited nodes
        if visited_nodes:
            visited = np.array(list(visited_nodes))
            ax.scatter(visited[:, 0], visited[:, 1], c='yellow', s=200,
                      edgecolors='orange', linewidth=2, zorder=3,
                      label='Visited', marker='o')

        # Draw pruned regions
        for node_point in pruned_nodes:
            # Find the node
            for step in kdtree.search_steps:
                if np.array_equal(step.node.point, node_point):
                    xmin, xmax, ymin, ymax = step.node.bounds
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                            linewidth=2, edgecolor='red',
                                            facecolor='red', alpha=0.1, zorder=1)
                    ax.add_patch(rect)
                    ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, 'PRUNED',
                           ha='center', va='center', fontsize=10, color='red',
                           weight='bold', alpha=0.7)
                    break

        # Current step highlighting
        current_step = kdtree.search_steps[step_idx]
        ax.scatter([current_step.node.point[0]], [current_step.node.point[1]],
                  c='lime', s=300, edgecolors='darkgreen', linewidth=3,
                  zorder=4, label='Current', marker='*')

    # Plot query point
    ax.scatter([query[0]], [query[1]], c='red', s=300, marker='X',
              edgecolors='darkred', linewidth=2, zorder=5, label='Query')

    # Draw circle for current best distance
    if len(kdtree.search_steps) > 0 and show_animation:
        current_step = kdtree.search_steps[step_idx]
        if current_step.best_dist < float('inf'):
            circle = plt.Circle(query, current_step.best_dist,
                              fill=False, edgecolor='blue', linestyle=':',
                              linewidth=2, label=f'Search radius')
            ax.add_patch(circle)

    # Find and highlight nearest neighbors (final result)
    neighbors = kdtree.search_knn(query, k_neighbors)
    if neighbors and not show_animation:
        neighbor_points = np.array([n[1] for n in neighbors])
        ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1],
                  c='lime', s=200, edgecolors='darkgreen', linewidth=2,
                  zorder=4, label=f'{k_neighbors}-NN')

        # Draw lines to neighbors
        for _, neighbor in neighbors:
            ax.plot([query[0], neighbor[0]], [query[1], neighbor[1]],
                   'g--', alpha=0.5, linewidth=1.5)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlabel('X', fontsize=12, weight='bold')
    ax.set_ylabel('Y', fontsize=12, weight='bold')
    ax.set_title('KD-Tree Spatial Partitioning', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    st.pyplot(fig)
    plt.close()

    st.caption("Click on the plot above to set a new query point (feature coming soon - use sidebar to set coordinates for now)")

with col2:
    st.subheader("Query Point")
    query_x = st.number_input("Query X", 0.0, 10.0, st.session_state.query_point[0], 0.1)
    query_y = st.number_input("Query Y", 0.0, 10.0, st.session_state.query_point[1], 0.1)

    if st.button("Update Query Point"):
        st.session_state.query_point = np.array([query_x, query_y])
        st.session_state.animation_step = 0
        st.rerun()

    # Search statistics
    st.markdown("---")
    st.subheader("Search Statistics")

    if len(kdtree.search_steps) > 0:
        total_steps = len(kdtree.search_steps)
        visited_count = sum(1 for s in kdtree.search_steps if s.action == 'visit')
        pruned_count = sum(1 for s in kdtree.search_steps if s.action == 'prune')
        backtrack_count = sum(1 for s in kdtree.search_steps if s.action == 'backtrack')

        st.metric("Total steps", total_steps)
        st.metric("Nodes visited", visited_count)
        st.metric("Branches pruned", pruned_count)
        st.metric("Backtracks", backtrack_count)

        efficiency = (1 - visited_count / len(points)) * 100
        st.metric("Pruning efficiency", f"{efficiency:.1f}%")

        # Show found neighbors
        st.markdown("---")
        st.subheader(f"Found {k_neighbors}-Nearest Neighbors")
        neighbors = kdtree.search_knn(query, k_neighbors)
        for i, (dist, point) in enumerate(neighbors):
            st.write(f"**#{i+1}**: Point ({point[0]:.2f}, {point[1]:.2f}) - Distance: {dist:.3f}")

# Step-by-step explanation
if show_animation and len(kdtree.search_steps) > 0:
    st.markdown("---")
    st.subheader("Current Step Explanation")

    current_step = kdtree.search_steps[step_idx]

    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"**Action**: {current_step.action.upper()}")
        st.write(f"**Node**: ({current_step.node.point[0]:.2f}, {current_step.node.point[1]:.2f})")
        st.write(f"**Depth**: {current_step.node.depth}")
        st.write(f"**Split axis**: {'X' if current_step.node.axis == 0 else 'Y'}")

    with col_b:
        st.write(f"**Distance to node**: {current_step.distance:.3f}")
        st.write(f"**Current best distance**: {current_step.best_dist:.3f}")
        if current_step.best_point is not None:
            st.write(f"**Best point so far**: ({current_step.best_point[0]:.2f}, {current_step.best_point[1]:.2f})")

    # Explanation text
    if current_step.action == 'visit':
        st.info(f"Visiting node at ({current_step.node.point[0]:.2f}, {current_step.node.point[1]:.2f}). Calculating distance to query point.")
    elif current_step.action == 'consider':
        st.success(f"Considering this point as a potential neighbor. Current best distance: {current_step.best_dist:.3f}")
    elif current_step.action == 'prune':
        st.warning(f"PRUNING this branch! The minimum distance to this region ({current_step.distance:.3f}) is greater than the current best distance ({current_step.best_dist:.3f}). No point in this branch can be closer.")
    elif current_step.action == 'backtrack':
        st.info(f"BACKTRACKING to explore the other branch. The other side might contain closer points because the minimum distance ({current_step.distance:.3f}) < best distance ({current_step.best_dist:.3f})")

# Educational content
st.markdown("---")
st.markdown("""
### How KD-Tree Search Works

1. **Start at root**: Begin at the top of the tree
2. **Recursive descent**: Go down the tree choosing the branch that contains the query point
3. **Update best**: Keep track of the closest point found so far
4. **Backtrack**: After reaching a leaf, go back up and check if the other branches might contain closer points
5. **Prune**: Skip branches where ALL points must be farther than the current best (this is the key optimization!)

### Pruning Condition

A branch can be pruned if:
```
minimum_distance_to_region > current_best_distance
```

This means even the CLOSEST possible point in that region is farther than what we've already found!

### Complexity

- **Brute force KNN**: O(N) - must check every point
- **KD-tree KNN**: O(log N) average case - prunes most branches
- **Worst case**: O(N) in high dimensions (curse of dimensionality)

### Try This

1. Move the query point to different locations
2. Watch how many branches get pruned
3. Notice how points near partition boundaries cause more backtracking
4. Increase K and see how it affects pruning efficiency
""")

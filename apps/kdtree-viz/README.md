# KD-Tree Backtracking & Pruning Visualization

Interactive Streamlit app demonstrating how KD-trees efficiently search for nearest neighbors through spatial partitioning, backtracking, and pruning.

## Features

- **Visual KD-Tree Partitioning**: See how space is recursively split
- **Step-by-Step Animation**: Watch the search algorithm in action
- **Pruning Visualization**: Highlighted regions that are skipped
- **Search Statistics**: Track visited nodes, pruned branches, and efficiency
- **Interactive Query**: Adjust query point and see results update

## Running the App

```bash
cd apps/kdtree-viz
streamlit run app.py
```

## How It Works

### KD-Tree Construction
- Points are recursively partitioned by alternating between X and Y axes
- Each node splits space with a hyperplane (line in 2D)
- Creates a binary tree structure for efficient searching

### Search with Backtracking & Pruning

1. **Descend**: Start at root, go down choosing the branch containing the query
2. **Update Best**: Track the closest point found so far
3. **Backtrack**: After reaching a leaf, go back up the tree
4. **Prune or Explore**: For each unexplored branch:
   - **Prune** if minimum distance to region > current best distance
   - **Explore** otherwise (backtrack into that branch)

### Why It's Fast

- **Brute Force**: O(N) - must check every point
- **KD-Tree Average**: O(log N) - prunes most branches
- **Pruning Efficiency**: Often skips 70-90% of points!

## Educational Use

Perfect for teaching:
- Spatial data structures
- Divide-and-conquer algorithms
- Nearest neighbor search optimization
- Backtracking strategies
- Branch-and-bound techniques

## Controls

- **Number of points**: Adjust dataset size
- **K neighbors**: How many neighbors to find
- **Random seed**: Change point distribution
- **Animation slider**: Step through search process
- **Query coordinates**: Set query point location

## Try This

1. Set query point near partition boundaries - more backtracking
2. Set query point in dense regions - more pruning
3. Increase K - watch how it affects pruning efficiency
4. Change random seed - see different tree structures

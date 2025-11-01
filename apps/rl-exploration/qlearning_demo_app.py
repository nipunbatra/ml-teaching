"""
Interactive Q-Learning Demo - Streamlit App

An educational, blog-post style interactive demonstration of Q-learning on MountainCar.
Shows code snippets, explanations, and live training visualization.

Run with: streamlit run qlearning_demo_app.py
"""

import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Q-Learning Interactive Demo",
    layout="wide",
    page_icon="🎓"
)

# =============================================================================
# Helper Functions
# =============================================================================

def discretize_state(state, pos_bins, vel_bins):
    """Convert continuous state to discrete indices"""
    pos, vel = state
    pos_idx = np.digitize(pos, pos_bins[1:-1])
    vel_idx = np.digitize(vel, vel_bins[1:-1])
    return pos_idx, vel_idx


def select_action_epsilon_greedy(q_table, state_idx, epsilon, n_actions):
    """Select action using epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q_table[state_idx])


def q_learning_update(q_table, state_idx, action, reward, next_state_idx, alpha, gamma):
    """Perform Q-learning update"""
    current_q = q_table[state_idx][action]
    max_next_q = np.max(q_table[next_state_idx])
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    q_table[state_idx][action] = new_q
    return new_q


# =============================================================================
# Title and Introduction
# =============================================================================

st.title("🎓 Interactive Q-Learning Tutorial")
st.markdown("### Learning to Solve MountainCar with Tabular Q-Learning")

st.markdown("""
Welcome to this interactive tutorial! You'll learn how **Q-learning** works by:
1. Understanding the MountainCar problem
2. Seeing the Q-learning algorithm in action
3. Training your own agent
4. Visualizing what the agent learned

Let's dive in! 🚀
""")

# =============================================================================
# Section 1: The Problem
# =============================================================================

st.header("1️⃣ The Problem: MountainCar")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **Goal:** Drive a car up a steep mountain by building momentum.

    **Challenge:** The car's engine isn't strong enough to drive straight up!

    **Solution:** Drive backwards first to build momentum, then forward.

    **State Space:**
    - Position: -1.2 (left) to 0.6 (right)
    - Velocity: -0.07 to 0.07

    **Actions:**
    - 0: Push left
    - 1: Do nothing
    - 2: Push right

    **Reward:** -1 for each timestep until goal (position ≥ 0.5)
    """)

with col2:
    # Environment details
    env = gym.make('MountainCar-v0')
    st.code(f"""
# Environment Details
Observation Space: {env.observation_space}
Action Space: {env.action_space}
Max Episode Steps: 200

# Sample observation
state = env.reset()
# Returns: [position, velocity]
# Example: [-0.5, 0.0]
    """, language='python')

# =============================================================================
# Section 2: Discretization
# =============================================================================

st.header("2️⃣ State Discretization")

st.markdown("""
**Problem:** Q-learning needs a **finite** state space, but MountainCar has **continuous** states!

**Solution:** **Discretize** the continuous state space into bins.
""")

n_bins_demo = st.slider("Number of bins per dimension", 3, 30, 10, key='bins_demo')

# Show discretization code
st.code(f"""
# Create bins for position and velocity
n_pos_bins = {n_bins_demo}
n_vel_bins = {n_bins_demo}

pos_bins = np.linspace(-1.2, 0.6, n_pos_bins)
vel_bins = np.linspace(-0.07, 0.07, n_vel_bins)

# Add infinity bounds for edge cases
pos_bins = np.concatenate([[-np.inf], pos_bins, [np.inf]])
vel_bins = np.concatenate([[-np.inf], vel_bins, [np.inf]])

# Discretize a state
def discretize_state(state):
    pos, vel = state
    pos_idx = np.digitize(pos, pos_bins[1:-1])
    vel_idx = np.digitize(vel, vel_bins[1:-1])
    return (pos_idx, vel_idx)

# Example
state = [-0.5, 0.02]  # continuous
state_idx = discretize_state(state)  # discrete: (7, 6)
""", language='python')

st.info(f"With {n_bins_demo} bins per dimension, we have **{n_bins_demo} × {n_bins_demo} = {n_bins_demo**2} discrete states** (instead of infinite!)")

# =============================================================================
# Section 3: Q-Table
# =============================================================================

st.header("3️⃣ The Q-Table")

st.markdown("""
The **Q-table** stores the expected future reward for each state-action pair:
- **Rows/Cols:** States (position bins × velocity bins)
- **Depth:** Actions (0, 1, 2)
- **Values:** Q(s, a) = expected future reward
""")

# Show Q-table structure
st.code(f"""
# Initialize Q-table
n_actions = 3
q_table = np.random.randn({n_bins_demo}, {n_bins_demo}, n_actions) * 0.1

# Q-table shape: ({n_bins_demo}, {n_bins_demo}, 3)
# Total entries: {n_bins_demo * n_bins_demo * 3}

# Access Q-values for a state
state_idx = (7, 6)  # discretized state
q_values = q_table[state_idx]
# Returns: [Q(s,left), Q(s,nothing), Q(s,right)]
# Example: [-0.02, 0.05, 0.03]

# Select best action
best_action = np.argmax(q_values)  # action with highest Q-value
""", language='python')

# Show actual initial Q-table
st.subheader("👀 See the Initial Q-Table")

st.markdown("""
Let's see what the Q-table looks like **before training**.
The values are small random numbers (initialized near zero).
""")

# Create a demo Q-table
demo_q_table = np.random.RandomState(42).randn(n_bins_demo, n_bins_demo, 3) * 0.1

# Create position and velocity bin labels
pos_bins_demo = np.linspace(-1.2, 0.6, n_bins_demo)
vel_bins_demo = np.linspace(-0.07, 0.07, n_bins_demo)

pos_labels_demo = []
for i in range(n_bins_demo - 1):
    low = pos_bins_demo[i]
    high = pos_bins_demo[i + 1]
    pos_labels_demo.append(f"P{i}:[{low:.2f},{high:.2f})")

vel_labels_demo = []
for i in range(n_bins_demo - 1):
    low = vel_bins_demo[i]
    high = vel_bins_demo[i + 1]
    vel_labels_demo.append(f"V{i}:[{low:.2f},{high:.2f})")

# Add one more to match the demo table size
pos_labels_demo.append(f"P{n_bins_demo-1}:[{pos_bins_demo[-1]:.2f},∞)")
vel_labels_demo.append(f"V{n_bins_demo-1}:[{vel_bins_demo[-1]:.2f},∞)")

# Create DataFrame for a sample
sample_indices = []
sample_values = []

# Sample evenly (show ~10-15 states)
sample_size = min(15, n_bins_demo * n_bins_demo)
step = max(1, (n_bins_demo * n_bins_demo) // sample_size)

idx = 0
for p_idx in range(n_bins_demo):
    for v_idx in range(n_bins_demo):
        if idx % step == 0 and len(sample_indices) < sample_size:
            sample_indices.append((pos_labels_demo[p_idx], vel_labels_demo[v_idx]))
            sample_values.append(demo_q_table[p_idx, v_idx, :])
        idx += 1

initial_q_df = pd.DataFrame(
    sample_values,
    index=pd.MultiIndex.from_tuples(sample_indices, names=['Position', 'Velocity']),
    columns=['Q(s, Left)', 'Q(s, Nothing)', 'Q(s, Right)']
)

initial_q_df['Best Action'] = initial_q_df.apply(
    lambda row: ['Left', 'Nothing', 'Right'][row.argmax()], axis=1
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Sample of Initial Q-Table:**")
    st.dataframe(
        initial_q_df.round(3).style.highlight_max(
            subset=['Q(s, Left)', 'Q(s, Nothing)', 'Q(s, Right)'],
            axis=1,
            color='lightyellow'
        ),
        use_container_width=True,
        height=350
    )

with col2:
    st.markdown("**Key Observations:**")
    st.markdown(f"""
    - Values are **random** and close to zero
    - Range: [{demo_q_table.min():.3f}, {demo_q_table.max():.3f}]
    - Mean: {demo_q_table.mean():.3f}
    - No clear pattern yet!

    **Best actions are random** because Q-values are random.

    After training, these values will become **meaningful** and show clear patterns!
    """)

# Show heatmap of initial Q-table
st.markdown("**Initial Q-Table Heatmaps (Before Training):**")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
action_names = ['Push Left (0)', 'Do Nothing (1)', 'Push Right (2)']

for i, ax in enumerate(axes):
    im = ax.imshow(demo_q_table[:, :, i].T, origin='lower', aspect='auto',
                  cmap='RdYlGn', vmin=-0.3, vmax=0.3)
    ax.set_title(f'Initial Q-Values: {action_names[i]}')
    ax.set_xlabel('Position Bin')
    ax.set_ylabel('Velocity Bin')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
st.pyplot(fig)

st.info("👆 Notice: Random noise, no clear pattern! Training will change this dramatically.")
plt.close()

# =============================================================================
# Section 4: Q-Learning Algorithm
# =============================================================================

st.header("4️⃣ The Q-Learning Algorithm")

st.markdown("""
Q-learning learns by **trial and error**, updating Q-values based on experience.
""")

st.subheader("The Q-Learning Update Rule")

st.latex(r"""
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
""")

st.markdown("""
Where:
- **Q(s, a)**: Current Q-value for state s, action a
- **α (alpha)**: Learning rate (0 to 1)
- **r**: Immediate reward
- **γ (gamma)**: Discount factor (0 to 1)
- **s'**: Next state
- **max Q(s', a')**: Best Q-value for next state
""")

# Interactive hyperparameters
col1, col2, col3 = st.columns(3)
with col1:
    alpha_demo = st.slider("α (Learning Rate)", 0.0, 1.0, 0.2, 0.05)
    st.caption("How much to update Q-values")
with col2:
    gamma_demo = st.slider("γ (Discount Factor)", 0.0, 1.0, 0.99, 0.01)
    st.caption("Importance of future rewards")
with col3:
    epsilon_demo = st.slider("ε (Exploration Rate)", 0.0, 1.0, 0.8, 0.05)
    st.caption("Probability of random action")

# Show algorithm code
st.code(f"""
# Q-Learning Training Loop
alpha = {alpha_demo}  # learning rate
gamma = {gamma_demo}  # discount factor
epsilon = {epsilon_demo}  # exploration rate

for episode in range(n_episodes):
    state = env.reset()

    for step in range(max_steps):
        # 1. Select action (epsilon-greedy)
        state_idx = discretize_state(state)
        if random() < epsilon:
            action = random_action()  # explore
        else:
            action = argmax(q_table[state_idx])  # exploit

        # 2. Take action, observe result
        next_state, reward, done = env.step(action)
        next_state_idx = discretize_state(next_state)

        # 3. Q-Learning update
        current_q = q_table[state_idx][action]
        max_next_q = max(q_table[next_state_idx])
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        q_table[state_idx][action] = new_q

        state = next_state
        if done:
            break

    # 4. Decay epsilon (explore less over time)
    epsilon *= 0.98
""", language='python')

# =============================================================================
# Section 5: Live Training
# =============================================================================

st.header("5️⃣ Train Your Own Agent!")

st.markdown("""
Now it's your turn! Configure the hyperparameters and watch the agent learn.
""")

# Training configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Configuration")

    n_pos_bins = st.slider("Position bins", 5, 40, 20, key='train_pos')
    n_vel_bins = st.slider("Velocity bins", 5, 40, 20, key='train_vel')

    alpha = st.slider("Learning rate (α)", 0.05, 1.0, 0.2, 0.05, key='train_alpha')
    gamma = st.slider("Discount factor (γ)", 0.8, 1.0, 0.99, 0.01, key='train_gamma')
    epsilon_start = st.slider("Initial exploration (ε)", 0.5, 1.0, 0.8, 0.05, key='train_eps')

    n_episodes = st.slider("Training episodes", 100, 2000, 500, 100)

with col2:
    st.subheader("📊 Expected Results")
    st.markdown(f"""
    **State space size:** {n_pos_bins} × {n_vel_bins} = **{n_pos_bins * n_vel_bins} states**

    **Q-table size:** {n_pos_bins * n_vel_bins * 3} entries

    **Training tips:**
    - More bins = more precision, slower learning
    - Higher α = faster learning, less stable
    - Higher γ = values long-term rewards more
    - Start with high ε, decay over time

    **What to expect:**
    - Episodes ~1-200: Random behavior, slow improvement
    - Episodes ~200-400: Starts reaching goal sometimes
    - Episodes ~400+: Consistent success, optimizing path
    """)

# Training button
if st.button("🚀 Start Training", type="primary", use_container_width=True):

    # Initialize
    env = gym.make('MountainCar-v0', render_mode=None)

    # Create bins
    pos_bins = np.linspace(-1.2, 0.6, n_pos_bins)
    vel_bins = np.linspace(-0.07, 0.07, n_vel_bins)
    pos_bins = np.concatenate([[-np.inf], pos_bins, [np.inf]])
    vel_bins = np.concatenate([[-np.inf], vel_bins, [np.inf]])

    # Initialize Q-table
    q_table = np.random.randn(n_pos_bins, n_vel_bins, 3) * 0.1

    # Training metrics
    rewards_history = []
    lengths_history = []
    epsilon = epsilon_start

    # Create placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    stats_placeholder = st.empty()

    # Training loop
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        episode_reward = 0

        for step in range(200):
            # Select action
            state_idx = discretize_state(state, pos_bins, vel_bins)
            action = select_action_epsilon_greedy(q_table, state_idx, epsilon, 3)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Q-learning update
            next_state_idx = discretize_state(next_state, pos_bins, vel_bins)
            q_learning_update(q_table, state_idx, action, reward, next_state_idx, alpha, gamma)

            state = next_state

            if terminated:
                break

        # Record metrics
        rewards_history.append(episode_reward)
        lengths_history.append(step + 1)

        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.98)

        # Update UI every 10 episodes
        if episode % 10 == 0 or episode == n_episodes - 1:
            progress_bar.progress((episode + 1) / n_episodes)
            status_text.text(f"Episode {episode + 1}/{n_episodes} | "
                           f"Reward: {episode_reward:.0f} | "
                           f"Steps: {step + 1} | "
                           f"ε: {epsilon:.3f}")

            # Plot progress
            if episode >= 20:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Rewards
                ax1.plot(rewards_history, alpha=0.3, label='Episode Reward')
                if len(rewards_history) >= 50:
                    rolling = pd.Series(rewards_history).rolling(50).mean()
                    ax1.plot(rolling, color='red', linewidth=2, label='50-Episode Avg')
                ax1.axhline(-200, color='green', linestyle='--', alpha=0.5, label='Success threshold')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Total Reward')
                ax1.set_title('Training Progress: Rewards')
                ax1.legend()
                ax1.grid(alpha=0.3)

                # Lengths
                ax2.plot(lengths_history, alpha=0.3, color='purple', label='Episode Length')
                if len(lengths_history) >= 50:
                    rolling = pd.Series(lengths_history).rolling(50).mean()
                    ax2.plot(rolling, color='orange', linewidth=2, label='50-Episode Avg')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Steps to Goal')
                ax2.set_title('Training Progress: Episode Length')
                ax2.legend()
                ax2.grid(alpha=0.3)

                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()

    env.close()

    # Final statistics
    st.success("✅ Training Complete!")

    col1, col2, col3, col4 = st.columns(4)

    recent_rewards = rewards_history[-100:]
    successes = sum(1 for r in recent_rewards if r > -200)

    with col1:
        st.metric("Final Avg Reward", f"{np.mean(recent_rewards):.1f}")
    with col2:
        st.metric("Success Rate", f"{successes}%")
    with col3:
        st.metric("Avg Steps", f"{np.mean(lengths_history[-100:]):.1f}")
    with col4:
        st.metric("Final ε", f"{epsilon:.3f}")

    # Store trained Q-table in session state
    st.session_state['q_table'] = q_table
    st.session_state['pos_bins'] = pos_bins
    st.session_state['vel_bins'] = vel_bins

    st.info("💡 Scroll down to see the Q-table visualization and test the agent!")

# =============================================================================
# Section 6: Q-Table Visualization
# =============================================================================

if 'q_table' in st.session_state:
    st.header("6️⃣ Visualizing the Learned Q-Table")

    st.markdown("""
    Let's see what the agent learned! Each heatmap shows Q-values for one action.
    - **Bright (green):** High Q-value = good action in that state
    - **Dark (red):** Low Q-value = bad action in that state
    """)

    q_table = st.session_state['q_table']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    action_names = ['Push Left (0)', 'Do Nothing (1)', 'Push Right (2)']

    for i, ax in enumerate(axes):
        im = ax.imshow(q_table[:, :, i].T, origin='lower', aspect='auto', cmap='RdYlGn')
        ax.set_title(f'Q-Values: {action_names[i]}')
        ax.set_xlabel('Position Bin')
        ax.set_ylabel('Velocity Bin')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**
    - **Bottom-left (low position, negative velocity):** "Push left" is good (build momentum going left)
    - **Top-right (high position, positive velocity):** "Push right" is good (push toward goal)
    - **Goal region (right side):** High Q-values overall (close to success!)
    """)

    # Show Q-table as dataframe
    st.subheader("📋 Q-Table as DataFrame")

    st.markdown("""
    Here's the actual Q-table data! Each row is a (position, velocity) state,
    and each column is an action's Q-value.
    """)

    # Create dataframe from Q-table
    pos_bins_display = st.session_state['pos_bins']
    vel_bins_display = st.session_state['vel_bins']

    # Create readable labels for state bins
    pos_labels = []
    for i in range(len(pos_bins_display) - 2):
        low = pos_bins_display[i+1]
        high = pos_bins_display[i+2]
        pos_labels.append(f"Pos {i}: [{low:.2f}, {high:.2f})")

    vel_labels = []
    for i in range(len(vel_bins_display) - 2):
        low = vel_bins_display[i+1]
        high = vel_bins_display[i+2]
        vel_labels.append(f"Vel {i}: [{low:.3f}, {high:.3f})")

    # Reshape Q-table for display
    n_pos = q_table.shape[0]
    n_vel = q_table.shape[1]

    # Create multi-index for better organization
    state_indices = []
    q_values_list = []

    for p_idx in range(n_pos):
        for v_idx in range(n_vel):
            state_indices.append((pos_labels[p_idx], vel_labels[v_idx]))
            q_values_list.append(q_table[p_idx, v_idx, :])

    # Create DataFrame
    q_df = pd.DataFrame(
        q_values_list,
        index=pd.MultiIndex.from_tuples(state_indices, names=['Position Bin', 'Velocity Bin']),
        columns=['Q(s, Left)', 'Q(s, Nothing)', 'Q(s, Right)']
    )

    # Add best action column
    q_df['Best Action'] = q_df.apply(lambda row: ['Left', 'Nothing', 'Right'][row.argmax()], axis=1)
    q_df['Max Q-Value'] = q_df[['Q(s, Left)', 'Q(s, Nothing)', 'Q(s, Right)']].max(axis=1)

    # Display options
    col1, col2 = st.columns(2)

    with col1:
        show_option = st.radio(
            "Display option:",
            ["Show All States", "Show High Q-Values Only", "Show Sample States"],
            index=2
        )

    with col2:
        n_decimals = st.slider("Decimal places:", 1, 4, 3, key='decimals')

    # Filter and display based on option
    if show_option == "Show High Q-Values Only":
        # Show states with high Q-values (likely important states)
        threshold = q_df['Max Q-Value'].quantile(0.75)
        display_df = q_df[q_df['Max Q-Value'] >= threshold].round(n_decimals)
        st.caption(f"Showing {len(display_df)} states with Max Q-Value ≥ {threshold:.3f} (top 25%)")
    elif show_option == "Show Sample States":
        # Show a sample of states
        sample_size = min(20, len(q_df))
        # Sample evenly across the state space
        step = len(q_df) // sample_size
        display_df = q_df.iloc[::step].round(n_decimals)
        st.caption(f"Showing {len(display_df)} sample states (evenly distributed)")
    else:
        # Show all
        display_df = q_df.round(n_decimals)
        st.caption(f"Showing all {len(display_df)} states (this may be slow for large Q-tables!)")

    # Display the dataframe with styling
    st.dataframe(
        display_df.style.highlight_max(
            subset=['Q(s, Left)', 'Q(s, Nothing)', 'Q(s, Right)'],
            axis=1,
            color='lightgreen'
        ),
        use_container_width=True,
        height=400
    )

    # Statistics
    st.subheader("📊 Q-Table Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total States", f"{n_pos * n_vel}")
    with col2:
        st.metric("Min Q-Value", f"{q_table.min():.3f}")
    with col3:
        st.metric("Max Q-Value", f"{q_table.max():.3f}")
    with col4:
        st.metric("Mean Q-Value", f"{q_table.mean():.3f}")

    # Action distribution
    st.subheader("🎯 Best Action Distribution")

    action_counts = q_df['Best Action'].value_counts()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        action_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Best Action for Each State')
        ax.set_xlabel('Action')
        ax.set_ylabel('Number of States')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Action Preferences:**")
        for action, count in action_counts.items():
            percentage = count / len(q_df) * 100
            st.write(f"- **{action}**: {count} states ({percentage:.1f}%)")

        st.markdown("""

        **What this tells us:**
        - The agent learned which actions work best in different states
        - "Left" is often best when far from goal (build momentum)
        - "Right" is often best when close to goal (push upward)
        - "Nothing" is rarely the best action (need to keep moving!)
        """)

# =============================================================================
# Section 7: Test the Agent
# =============================================================================

if 'q_table' in st.session_state:
    st.header("7️⃣ See the Trained Agent in Action!")

    st.markdown("Let's watch the trained agent solve MountainCar using the learned Q-table.")

    if st.button("▶️ Run Test Episode", type="primary"):
        env = gym.make('MountainCar-v0', render_mode='rgb_array')
        state, _ = env.reset(seed=42)

        q_table = st.session_state['q_table']
        pos_bins = st.session_state['pos_bins']
        vel_bins = st.session_state['vel_bins']

        trajectory = []

        for step in range(200):
            state_idx = discretize_state(state, pos_bins, vel_bins)
            action = np.argmax(q_table[state_idx])

            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectory.append({
                'step': step,
                'position': state[0],
                'velocity': state[1],
                'action': ['Left', 'Nothing', 'Right'][action],
                'reward': reward
            })

            state = next_state

            if terminated:
                st.success(f"🎉 Goal reached in {step + 1} steps!")
                break

        env.close()

        # Plot trajectory
        df = pd.DataFrame(trajectory)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Position over time
        ax1.plot(df['step'], df['position'], marker='o', markersize=3)
        ax1.axhline(0.5, color='green', linestyle='--', label='Goal')
        ax1.set_ylabel('Position')
        ax1.set_title('Agent Trajectory')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Velocity over time
        ax2.plot(df['step'], df['velocity'], marker='o', markersize=3, color='orange')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Show action sequence
        st.subheader("Action Sequence")
        st.dataframe(df, use_container_width=True)

# =============================================================================
# Section 8: Key Takeaways
# =============================================================================

st.header("8️⃣ Key Takeaways")

st.markdown("""
### What You Learned

1. **State Discretization**: Converting continuous states to discrete bins
2. **Q-Table**: Storing expected rewards for state-action pairs
3. **Q-Learning Update**: The core learning algorithm
4. **Exploration vs Exploitation**: Balancing trying new things vs using what works
5. **Hyperparameters**: α, γ, ε and their effects on learning

### Limitations of Tabular Q-Learning

- **Scalability**: Doesn't work for high-dimensional states (images, etc.)
- **Discretization**: Loss of precision from continuous → discrete
- **Curse of Dimensionality**: State space grows exponentially

### Next Steps

To handle complex problems (like Atari games), we use **Deep Q-Networks (DQN)**:
- Replace Q-table with a neural network
- Can handle high-dimensional inputs (images!)
- No discretization needed

Check out the DQN demo in `apps/rl-demo/` to see this in action! 🚀
""")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
### 📚 Further Reading

- [Sutton & Barto - Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up - Q-Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)

### 🔗 Related Demos

- **Simple Heuristics**: `apps/rl-exploration/cartpole-simple-demo.py`
- **DQN Training**: `apps/rl-demo/`
- **FlappyBird RL**: `apps/flappy-rl-demo/`

---

**Created for ML Teaching** | [GitHub](https://github.com/nipunbatra/ml-teaching)
""")

#!/usr/bin/env python3
"""
Q-Learning on MountainCar-v0

Demonstrates tabular Q-learning with discretized state space:
- Train Q-table from scratch
- Save/load trained Q-tables
- Visualize learning progress
- Show trained agent in action

This is a great introduction to Q-learning before moving to Deep Q-Networks (DQN).

Run with: python qlearning-mountaincar.py [command]
"""

import gymnasium as gym
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import time


class QLearningAgent:
    """
    Q-Learning agent with discretized state space
    """

    def __init__(self, n_pos_bins=25, n_vel_bins=25, n_actions=3,
                 alpha=0.2, gamma=0.99, epsilon=0.8):
        """
        Args:
            n_pos_bins: Number of bins for position discretization
            n_vel_bins: Number of bins for velocity discretization
            n_actions: Number of actions (3 for MountainCar)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
        """
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98

        # Create bins for discretization
        self.pos_bins = np.linspace(-1.2, 0.6, n_pos_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, n_vel_bins)

        # Add infinity bounds
        self.pos_bins = np.concatenate([[-np.inf], self.pos_bins, [np.inf]])
        self.vel_bins = np.concatenate([[-np.inf], self.vel_bins, [np.inf]])

        # Initialize Q-table with small random values
        self.q_table = np.random.randn(n_pos_bins, n_vel_bins, n_actions) * 0.1

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []

    def discretize_state(self, state):
        """Convert continuous state to discrete indices"""
        pos, vel = state
        pos_idx = np.digitize(pos, self.pos_bins[1:-1])
        vel_idx = np.digitize(vel, self.vel_bins[1:-1])
        return pos_idx, vel_idx

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        pos_idx, vel_idx = self.discretize_state(state)

        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[pos_idx, vel_idx])

    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule"""
        pos_idx, vel_idx = self.discretize_state(state)
        next_pos_idx, next_vel_idx = self.discretize_state(next_state)

        # Q-learning update
        current_q = self.q_table[pos_idx, vel_idx, action]
        max_next_q = np.max(self.q_table[next_pos_idx, next_vel_idx])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        self.q_table[pos_idx, vel_idx, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table and parameters"""
        data = {
            'q_table': self.q_table,
            'n_pos_bins': self.n_pos_bins,
            'n_vel_bins': self.n_vel_bins,
            'n_actions': self.n_actions,
            'pos_bins': self.pos_bins,
            'vel_bins': self.vel_bins,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Q-table to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load Q-table and parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        agent = cls(
            n_pos_bins=data['n_pos_bins'],
            n_vel_bins=data['n_vel_bins'],
            n_actions=data['n_actions'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon=data['epsilon']
        )
        agent.q_table = data['q_table']
        agent.pos_bins = data['pos_bins']
        agent.vel_bins = data['vel_bins']
        agent.episode_rewards = data.get('episode_rewards', [])
        agent.episode_lengths = data.get('episode_lengths', [])

        print(f"Loaded Q-table from {filepath}")
        return agent


def train(n_episodes=4000, n_steps=200, save_path='qtable_mountaincar.pkl',
          render_every=None, verbose=True):
    """
    Train Q-learning agent on MountainCar

    Args:
        n_episodes: Number of training episodes
        n_steps: Maximum steps per episode
        save_path: Where to save the trained Q-table
        render_every: Render every N episodes (None = no rendering)
        verbose: Print progress
    """

    env = gym.make('MountainCar-v0', render_mode=None)
    agent = QLearningAgent()

    if verbose:
        print("\n" + "="*70)
        print("TRAINING Q-LEARNING AGENT ON MOUNTAINCAR-V0")
        print("="*70)
        print(f"Episodes: {n_episodes}")
        print(f"Max steps per episode: {n_steps}")
        print(f"State bins: {agent.n_pos_bins} x {agent.n_vel_bins} = {agent.n_pos_bins * agent.n_vel_bins}")
        print(f"Q-table shape: {agent.q_table.shape}")
        print(f"Learning rate (alpha): {agent.alpha}")
        print(f"Discount factor (gamma): {agent.gamma}")
        print(f"Initial epsilon: {agent.epsilon}")
        print("="*70 + "\n")

    successful_episodes = 0

    for episode in range(n_episodes):
        # Reset environment
        state, _ = env.reset(seed=episode)
        episode_reward = 0

        # Run episode
        for step in range(n_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            if terminated:
                successful_episodes += 1
                if verbose and episode % 100 == 0:
                    print(f"Episode {episode}: SUCCESS in {step} steps! "
                          f"(Total successes: {successful_episodes})")
                break

        # Store metrics
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(step + 1)

        # Decay epsilon
        agent.decay_epsilon()

        # Progress report
        if verbose and (episode + 1) % 100 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            recent_lengths = agent.episode_lengths[-100:]
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg reward (last 100): {np.mean(recent_rewards):.2f}")
            print(f"  Avg length (last 100): {np.mean(recent_lengths):.2f}")
            print(f"  Success rate (last 100): {sum(1 for r in recent_rewards if r > -200)/len(recent_rewards)*100:.1f}%")

    env.close()

    # Save trained agent
    agent.save(save_path)

    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total successful episodes: {successful_episodes}/{n_episodes} "
              f"({successful_episodes/n_episodes*100:.1f}%)")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Saved to: {save_path}")
        print("="*70 + "\n")

    return agent


def test(qtable_path='qtable_mountaincar.pkl', n_episodes=5, render=True, verbose=True):
    """
    Test trained Q-learning agent

    Args:
        qtable_path: Path to trained Q-table
        n_episodes: Number of test episodes
        render: Whether to render
        verbose: Print progress
    """

    # Load agent
    agent = QLearningAgent.load(qtable_path)

    # Create environment
    render_mode = 'human' if render else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)

    if verbose:
        print("\n" + "="*70)
        print("TESTING TRAINED AGENT")
        print("="*70)
        print(f"Loaded Q-table from: {qtable_path}")
        print(f"Running {n_episodes} test episodes...")
        print("="*70 + "\n")

    test_rewards = []
    test_lengths = []
    successes = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=42 + episode)
        episode_reward = 0

        for step in range(300):
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = next_state

            if render:
                time.sleep(0.01)

            if terminated:
                successes += 1
                if verbose:
                    print(f"Episode {episode + 1}: SUCCESS in {step} steps! Reward: {episode_reward:.1f}")
                break

        if not terminated and verbose:
            print(f"Episode {episode + 1}: Did not reach goal in {step} steps. Reward: {episode_reward:.1f}")

        test_rewards.append(episode_reward)
        test_lengths.append(step + 1)

    env.close()

    if verbose:
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Success rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
        print(f"Average reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        print(f"Average steps: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
        print(f"Best reward: {np.max(test_rewards):.1f}")
        print("="*70 + "\n")

    return test_rewards, test_lengths


def visualize_training(qtable_path='qtable_mountaincar.pkl', save_fig=None):
    """Visualize training progress"""

    agent = QLearningAgent.load(qtable_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot rewards
    ax = axes[0]
    ax.plot(agent.episode_rewards, alpha=0.3, label='Episode Reward')

    # Rolling average
    window = 100
    if len(agent.episode_rewards) >= window:
        rolling_mean = np.convolve(agent.episode_rewards,
                                    np.ones(window)/window,
                                    mode='valid')
        ax.plot(range(window-1, len(agent.episode_rewards)),
               rolling_mean,
               color='red',
               linewidth=2,
               label=f'{window}-Episode Average')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress: Rewards')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot episode lengths
    ax = axes[1]
    ax.plot(agent.episode_lengths, alpha=0.3, label='Episode Length', color='green')

    if len(agent.episode_lengths) >= window:
        rolling_mean = np.convolve(agent.episode_lengths,
                                    np.ones(window)/window,
                                    mode='valid')
        ax.plot(range(window-1, len(agent.episode_lengths)),
               rolling_mean,
               color='orange',
               linewidth=2,
               label=f'{window}-Episode Average')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Goal')
    ax.set_title('Training Progress: Episode Length')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()


def visualize_qtable(qtable_path='qtable_mountaincar.pkl', save_fig=None):
    """Visualize learned Q-table as heatmaps"""

    agent = QLearningAgent.load(qtable_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    action_names = ['Push Left', 'Do Nothing', 'Push Right']

    for action in range(3):
        ax = axes[action]
        q_values = agent.q_table[:, :, action]

        im = ax.imshow(q_values.T, origin='lower', aspect='auto', cmap='RdYlGn')
        ax.set_title(f'Q-Values for Action: {action_names[action]}')
        ax.set_xlabel('Position Bin')
        ax.set_ylabel('Velocity Bin')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()


def show_info(qtable_path='qtable_mountaincar.pkl'):
    """Show information about a trained Q-table"""

    agent = QLearningAgent.load(qtable_path)

    print("\n" + "="*70)
    print("Q-TABLE INFORMATION")
    print("="*70)
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"Position bins: {agent.n_pos_bins}")
    print(f"Velocity bins: {agent.n_vel_bins}")
    print(f"Actions: {agent.n_actions}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate (alpha): {agent.alpha}")
    print(f"  Discount factor (gamma): {agent.gamma}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"\nTraining history:")
    print(f"  Episodes trained: {len(agent.episode_rewards)}")
    if agent.episode_rewards:
        print(f"  Final avg reward (last 100): {np.mean(agent.episode_rewards[-100:]):.2f}")
        print(f"  Final avg length (last 100): {np.mean(agent.episode_lengths[-100:]):.2f}")
    print(f"\nQ-value statistics:")
    print(f"  Min: {agent.q_table.min():.4f}")
    print(f"  Max: {agent.q_table.max():.4f}")
    print(f"  Mean: {agent.q_table.mean():.4f}")
    print(f"  Std: {agent.q_table.std():.4f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Q-Learning on MountainCar-v0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent
  python qlearning-mountaincar.py train --episodes 5000

  # Test trained agent
  python qlearning-mountaincar.py test

  # Visualize training progress
  python qlearning-mountaincar.py plot-training

  # Visualize Q-table heatmaps
  python qlearning-mountaincar.py plot-qtable

  # Show Q-table info
  python qlearning-mountaincar.py info

Teaching Notes:
  This demonstrates tabular Q-learning with discretized states.
  Great for understanding Q-learning before Deep Q-Networks (DQN).

  Key concepts:
  - State discretization (continuous → discrete)
  - Q-learning update rule
  - Exploration vs exploitation (epsilon-greedy)
  - Convergence and learning curves
        """
    )

    parser.add_argument(
        'command',
        choices=['train', 'test', 'plot-training', 'plot-qtable', 'info'],
        help='Command to execute'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=4000,
        help='Number of training episodes (default: 4000)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Maximum steps per episode (default: 200)'
    )

    parser.add_argument(
        '--qtable',
        type=str,
        default='qtable_mountaincar.pkl',
        help='Path to Q-table file (default: qtable_mountaincar.pkl)'
    )

    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering for test'
    )

    parser.add_argument(
        '--save-fig',
        type=str,
        help='Save figure to this path instead of showing'
    )

    parser.add_argument(
        '--test-episodes',
        type=int,
        default=5,
        help='Number of test episodes (default: 5)'
    )

    args = parser.parse_args()

    if args.command == 'train':
        train(
            n_episodes=args.episodes,
            n_steps=args.steps,
            save_path=args.qtable
        )

    elif args.command == 'test':
        test(
            qtable_path=args.qtable,
            n_episodes=args.test_episodes,
            render=not args.no_render
        )

    elif args.command == 'plot-training':
        visualize_training(
            qtable_path=args.qtable,
            save_fig=args.save_fig
        )

    elif args.command == 'plot-qtable':
        visualize_qtable(
            qtable_path=args.qtable,
            save_fig=args.save_fig
        )

    elif args.command == 'info':
        show_info(qtable_path=args.qtable)


if __name__ == '__main__':
    main()

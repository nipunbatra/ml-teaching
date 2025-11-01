#!/usr/bin/env python3
"""
CartPole Simple Demo

Demonstrates CartPole environment with simple strategies:
1. Random actions (baseline)
2. Simple heuristic (based on pole angle)
3. Position-based heuristic
4. Combined heuristic

Perfect for teaching RL basics without complex neural networks.

Run with: python cartpole-simple-demo.py [strategy]
"""

import gymnasium as gym
import numpy as np
import argparse
import time


def random_strategy(observation, env):
    """Random action selection (baseline)"""
    return env.action_space.sample()


def angle_heuristic(observation, env):
    """
    Simple rule: if pole is falling right, push right; if falling left, push left

    observation = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    action: 0 = push left, 1 = push right
    """
    pole_angle = observation[2]

    # If pole tilting right (positive angle), push right
    # If pole tilting left (negative angle), push left
    if pole_angle > 0:
        return 1  # push right
    else:
        return 0  # push left


def velocity_heuristic(observation, env):
    """Use pole angular velocity for faster reaction"""
    pole_velocity = observation[3]

    # React to velocity, not just position
    if pole_velocity > 0:
        return 1  # push right
    else:
        return 0  # push left


def combined_heuristic(observation, env):
    """
    Combine angle and velocity for better performance
    Also consider cart position to avoid edges
    """
    cart_position = observation[0]
    cart_velocity = observation[1]
    pole_angle = observation[2]
    pole_velocity = observation[3]

    # Combined score: weighted sum of angle and velocity
    # Velocity is more predictive, so weight it higher
    score = pole_angle + 0.5 * pole_velocity

    # Adjust for cart position to avoid edges
    # If cart is too far right and moving right, push left
    if cart_position > 2.0 and cart_velocity > 0:
        return 0  # push left
    # If cart is too far left and moving left, push right
    elif cart_position < -2.0 and cart_velocity < 0:
        return 1  # push right

    # Otherwise, follow the combined score
    if score > 0:
        return 1  # push right
    else:
        return 0  # push left


def run_episode(strategy_name, max_steps=500, render=True, verbose=True):
    """Run one episode with the given strategy"""

    # Create environment
    render_mode = 'human' if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)

    observation, info = env.reset(seed=42)

    total_reward = 0
    step = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy_name.upper()}")
        print(f"{'='*70}")
        print(f"Initial observation: {observation}")
        print(f"\n{'Step':<6} {'Reward':<8} {'Total':<8} {'Angle':<10} {'Position':<10}")
        print(f"{'-'*70}")

    while step < max_steps:
        # Select action based on strategy
        if strategy_name == 'random':
            action = random_strategy(observation, env)
        elif strategy_name == 'angle':
            action = angle_heuristic(observation, env)
        elif strategy_name == 'velocity':
            action = velocity_heuristic(observation, env)
        elif strategy_name == 'combined':
            action = combined_heuristic(observation, env)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Print progress
        if verbose and step % 50 == 0:
            print(f"{step:<6} {reward:<8.1f} {total_reward:<8.1f} "
                  f"{observation[2]:<10.4f} {observation[0]:<10.4f}")

        # Render at reasonable speed
        if render:
            time.sleep(0.02)

        # Check if done
        if terminated or truncated:
            if verbose:
                print(f"\n{'='*70}")
                if total_reward >= 500:
                    print(f"🎉 PERFECT! Balanced for maximum {step} steps!")
                elif total_reward >= 200:
                    print(f"✓ Good! Balanced for {step} steps")
                else:
                    print(f"Pole fell after {step} steps")
                print(f"Total reward: {total_reward:.1f}")
                print(f"{'='*70}\n")
            break

    if step >= max_steps and verbose:
        print(f"\n{'='*70}")
        print(f"🏆 Completed all {max_steps} steps!")
        print(f"Total reward: {total_reward:.1f}")
        print(f"{'='*70}\n")

    env.close()
    return total_reward, step


def compare_strategies(num_trials=10):
    """Compare all strategies without rendering"""
    strategies = ['random', 'angle', 'velocity', 'combined']

    print("\n" + "="*70)
    print("COMPARING STRATEGIES")
    print("="*70)
    print(f"Running {num_trials} trials for each strategy...\n")

    results = {}

    for strategy in strategies:
        print(f"Testing {strategy.upper()} strategy...")
        rewards = []

        for trial in range(num_trials):
            reward, steps = run_episode(
                strategy,
                max_steps=500,
                render=False,
                verbose=False
            )
            rewards.append(reward)

        results[strategy] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        }

    # Print results table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Strategy':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*70)

    for strategy, metrics in results.items():
        print(f"{strategy.upper():<15} "
              f"{metrics['mean']:<12.1f} "
              f"{metrics['std']:<12.1f} "
              f"{metrics['min']:<12.1f} "
              f"{metrics['max']:<12.1f}")

    print("="*70)

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"\n🏆 Best strategy: {best_strategy[0].upper()} "
          f"(mean reward: {best_strategy[1]['mean']:.1f})")

    # Interpretation
    print("\n📊 Interpretation:")
    print("  - Random: Baseline performance (~20-30)")
    print("  - Angle: Basic reactive control (~40-80)")
    print("  - Velocity: Faster reaction (~80-120)")
    print("  - Combined: Best simple heuristic (~100-200)")
    print("  - Note: DQN can achieve 400-500 with learning!")
    print()


def show_environment_info():
    """Display information about the CartPole environment"""
    env = gym.make('CartPole-v1')

    print("\n" + "="*70)
    print("CARTPOLE-V1 ENVIRONMENT INFO")
    print("="*70)

    print("\nObservation Space:")
    print(f"  Type: {env.observation_space}")
    print(f"  Low:  {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    print(f"  Shape: {env.observation_space.shape}")

    print("\nObservation Variables:")
    print("  [0] Cart Position:       -4.8 to 4.8")
    print("  [1] Cart Velocity:       -Inf to Inf")
    print("  [2] Pole Angle:          -0.418 to 0.418 radians (~24°)")
    print("  [3] Pole Angular Velocity: -Inf to Inf")

    print("\nAction Space:")
    print(f"  Type: {env.action_space}")
    print(f"  Actions: {env.action_space.n}")
    print("  0 = Push cart to the left")
    print("  1 = Push cart to the right")

    print("\nReward:")
    print("  +1 for each timestep the pole stays upright")
    print("  Maximum episode length: 500 steps")

    print("\nTermination Conditions:")
    print("  - Pole angle > 12° from vertical")
    print("  - Cart position > 2.4 units from center")
    print("  - Episode length > 500 steps (success!)")

    print("\nSuccess Criteria:")
    print("  Average reward ≥ 475 over 100 consecutive trials")

    print("="*70 + "\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='CartPole strategy demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cartpole-simple-demo.py random      # Random baseline
  python cartpole-simple-demo.py angle       # Angle-based heuristic
  python cartpole-simple-demo.py velocity    # Velocity-based heuristic
  python cartpole-simple-demo.py combined    # Best simple heuristic
  python cartpole-simple-demo.py compare     # Compare all strategies
  python cartpole-simple-demo.py info        # Show environment info

Teaching Notes:
  This demo shows that simple heuristics can solve CartPole partially,
  but reinforcement learning (DQN) achieves near-perfect performance.
  Use this to motivate why we need learning algorithms!
        """
    )

    parser.add_argument(
        'strategy',
        choices=['random', 'angle', 'velocity', 'combined', 'compare', 'info'],
        help='Strategy to use or action to perform'
    )

    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (faster)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Maximum steps per episode (default: 500)'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials for compare mode (default: 10)'
    )

    args = parser.parse_args()

    if args.strategy == 'info':
        show_environment_info()
    elif args.strategy == 'compare':
        compare_strategies(num_trials=args.trials)
    else:
        run_episode(
            args.strategy,
            max_steps=args.steps,
            render=not args.no_render
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MountainCar-v0 Demo with Different Strategies

Demonstrates various approaches to solving the MountainCar environment:
1. Random actions
2. Always go right
3. Build momentum (go left first, then right)
4. Simple heuristic based on velocity

Run with: python mountaincar-demo.py [strategy]
Strategies: random, right, momentum, heuristic
"""

import gymnasium as gym
import numpy as np
import argparse
import time


def random_strategy(observation, env):
    """Random action selection"""
    return env.action_space.sample()


def always_right_strategy(observation, env):
    """Always push right"""
    return 2  # 0=left, 1=nothing, 2=right


def momentum_strategy(observation, env, step, switch_step=50):
    """Go left first to build momentum, then go right"""
    return 0 if step < switch_step else 2


def heuristic_strategy(observation, env):
    """
    Simple heuristic based on position and velocity
    observation = [position, velocity]
    position: -1.2 to 0.6 (goal is at 0.5)
    velocity: -0.07 to 0.07
    """
    position, velocity = observation

    # If moving right and close to goal, keep going right
    if velocity > 0:
        return 2  # push right
    # If moving left, push left to build momentum
    else:
        return 0  # push left


def run_episode(strategy_name, max_steps=250, render=True, verbose=True):
    """Run one episode with the given strategy"""

    # Create environment
    render_mode = 'human' if render else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)

    observation, info = env.reset(seed=42)

    total_reward = 0
    step = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name.upper()}")
        print(f"{'='*60}")
        print(f"{'Step':<6} {'Position':<12} {'Velocity':<12} {'Action':<10} {'Reward':<8}")
        print(f"{'-'*60}")

    while step < max_steps:
        # Select action based on strategy
        if strategy_name == 'random':
            action = random_strategy(observation, env)
        elif strategy_name == 'right':
            action = always_right_strategy(observation, env)
        elif strategy_name == 'momentum':
            action = momentum_strategy(observation, env, step)
        elif strategy_name == 'heuristic':
            action = heuristic_strategy(observation, env)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Print progress
        if verbose and step % 10 == 0:
            action_name = ['LEFT', 'NOTHING', 'RIGHT'][action]
            print(f"{step:<6} {observation[0]:<12.6f} {observation[1]:<12.6f} "
                  f"{action_name:<10} {reward:<8.1f}")

        # Render at reasonable speed
        if render:
            time.sleep(0.01)

        # Check if done
        if terminated:
            if verbose:
                print(f"\n{'='*60}")
                print(f"🎉 SUCCESS! Reached the goal in {step} steps!")
                print(f"Final position: {observation[0]:.6f}")
                print(f"Total reward: {total_reward:.1f}")
                print(f"{'='*60}\n")
            break

    if not terminated and verbose:
        print(f"\n{'='*60}")
        print(f"❌ Did not reach goal in {max_steps} steps")
        print(f"Final position: {observation[0]:.6f}")
        print(f"Total reward: {total_reward:.1f}")
        print(f"{'='*60}\n")

    env.close()
    return total_reward, step, terminated


def compare_strategies(num_trials=5):
    """Compare all strategies without rendering"""
    strategies = ['random', 'right', 'momentum', 'heuristic']

    print("\n" + "="*70)
    print("COMPARING STRATEGIES")
    print("="*70)
    print(f"Running {num_trials} trials for each strategy...\n")

    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} strategy...")
        rewards = []
        steps_list = []
        successes = 0

        for trial in range(num_trials):
            reward, steps, success = run_episode(
                strategy,
                max_steps=250,
                render=False,
                verbose=False
            )
            rewards.append(reward)
            steps_list.append(steps)
            if success:
                successes += 1

        results[strategy] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'success_rate': successes / num_trials * 100
        }

    # Print results table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Strategy':<15} {'Mean Reward':<15} {'Mean Steps':<15} {'Success Rate':<15}")
    print("-"*70)

    for strategy, metrics in results.items():
        print(f"{strategy.upper():<15} "
              f"{metrics['mean_reward']:<15.1f} "
              f"{metrics['mean_steps']:<15.1f} "
              f"{metrics['success_rate']:<15.0f}%")

    print("="*70)

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['success_rate'])
    print(f"\n🏆 Best strategy: {best_strategy[0].upper()} "
          f"({best_strategy[1]['success_rate']:.0f}% success rate)")
    print()


def show_environment_info():
    """Display information about the MountainCar environment"""
    env = gym.make('MountainCar-v0')

    print("\n" + "="*70)
    print("MOUNTAINCAR-V0 ENVIRONMENT INFO")
    print("="*70)

    print("\nObservation Space:")
    print(f"  Type: {env.observation_space}")
    print(f"  Low:  {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    print(f"  Shape: {env.observation_space.shape}")

    print("\nObservation Variables:")
    print("  [0] Position:  -1.2 (left) to 0.6 (right), goal at 0.5")
    print("  [1] Velocity:  -0.07 to 0.07")

    print("\nAction Space:")
    print(f"  Type: {env.action_space}")
    print(f"  Actions: {env.action_space.n}")
    print("  0 = Push left")
    print("  1 = Do nothing")
    print("  2 = Push right")

    print("\nReward:")
    print("  -1 for each timestep until goal is reached")
    print("  Goal: Reach position ≥ 0.5")

    print("\nTermination:")
    print("  Episode ends when position ≥ 0.5 (goal reached)")

    print("="*70 + "\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='MountainCar-v0 strategy demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mountaincar-demo.py random         # Random actions
  python mountaincar-demo.py right          # Always go right
  python mountaincar-demo.py momentum       # Build momentum strategy
  python mountaincar-demo.py heuristic      # Velocity-based heuristic
  python mountaincar-demo.py compare        # Compare all strategies
  python mountaincar-demo.py info           # Show environment info
        """
    )

    parser.add_argument(
        'strategy',
        choices=['random', 'right', 'momentum', 'heuristic', 'compare', 'info'],
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
        default=250,
        help='Maximum steps per episode (default: 250)'
    )

    args = parser.parse_args()

    if args.strategy == 'info':
        show_environment_info()
    elif args.strategy == 'compare':
        compare_strategies()
    else:
        run_episode(
            args.strategy,
            max_steps=args.steps,
            render=not args.no_render
        )


if __name__ == '__main__':
    main()

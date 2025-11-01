#!/usr/bin/env python3
"""
Gymnasium Environments Explorer

Browse and test different Gymnasium environments.
Great for understanding action spaces, observation spaces, and basic interaction.

Run with: python gym-explorer.py [environment_name]
"""

import gymnasium as gym
import argparse
import numpy as np
import time


def list_environments():
    """Display all available environments"""
    print("\n" + "="*70)
    print("AVAILABLE GYMNASIUM ENVIRONMENTS")
    print("="*70 + "\n")

    all_envs = list(gym.envs.registry.keys())

    categories = {
        'Classic Control': [],
        'Box2D': [],
        'MuJoCo': [],
        'Toy Text': [],
        'Atari': [],
        'Other': []
    }

    # Categorize environments
    for env_name in all_envs:
        if any(x in env_name for x in ['CartPole', 'MountainCar', 'Pendulum', 'Acrobot']):
            categories['Classic Control'].append(env_name)
        elif any(x in env_name for x in ['Lunar', 'Bipedal', 'CarRacing']):
            categories['Box2D'].append(env_name)
        elif any(x in env_name for x in ['Reacher', 'Pusher', 'Hopper', 'Walker', 'Ant', 'Humanoid', 'HalfCheetah', 'Swimmer', 'InvertedPendulum', 'InvertedDoublePendulum']):
            categories['MuJoCo'].append(env_name)
        elif any(x in env_name for x in ['Blackjack', 'FrozenLake', 'CliffWalking', 'Taxi']):
            categories['Toy Text'].append(env_name)
        elif 'ALE' in env_name or 'atari' in env_name.lower():
            categories['Atari'].append(env_name)
        else:
            categories['Other'].append(env_name)

    # Print categorized list
    for category, envs in categories.items():
        if envs:
            print(f"\n{category} ({len(envs)} environments):")
            print("-" * 70)
            for i, env in enumerate(sorted(envs), 1):
                print(f"  {i:2}. {env}")

    print("\n" + "="*70)
    print(f"Total: {len(all_envs)} environments")
    print("="*70 + "\n")


def show_env_info(env_name):
    """Display detailed information about an environment"""
    try:
        # Try creating without render mode first (some envs don't support it)
        try:
            env = gym.make(env_name)
        except:
            env = gym.make(env_name, render_mode=None)

        print("\n" + "="*70)
        print(f"ENVIRONMENT: {env_name}")
        print("="*70)

        # Observation space
        print("\n📊 OBSERVATION SPACE:")
        print(f"  Type: {type(env.observation_space).__name__}")
        print(f"  Details: {env.observation_space}")

        if hasattr(env.observation_space, 'low'):
            print(f"  Low:  {env.observation_space.low}")
            print(f"  High: {env.observation_space.high}")
            print(f"  Shape: {env.observation_space.shape}")

        if hasattr(env.observation_space, 'n'):
            print(f"  Discrete values: {env.observation_space.n}")

        # Action space
        print("\n🎮 ACTION SPACE:")
        print(f"  Type: {type(env.action_space).__name__}")
        print(f"  Details: {env.action_space}")

        if hasattr(env.action_space, 'n'):
            print(f"  Number of actions: {env.action_space.n}")
            print("\n  Sample actions:")
            for i in range(min(10, env.action_space.n)):
                print(f"    {i}: {env.action_space.sample()}")

        if hasattr(env.action_space, 'low'):
            print(f"  Low:  {env.action_space.low}")
            print(f"  High: {env.action_space.high}")
            print(f"  Shape: {env.action_space.shape}")

        # Sample observations
        print("\n🔍 SAMPLE OBSERVATIONS (5 random resets):")
        for i in range(5):
            obs, info = env.reset()
            print(f"  {i+1}. {obs}")

        # Reward range
        if hasattr(env, 'reward_range'):
            print(f"\n💰 REWARD RANGE: {env.reward_range}")

        # Max episode steps
        if hasattr(env, '_max_episode_steps'):
            print(f"\n⏱️  MAX EPISODE STEPS: {env._max_episode_steps}")

        print("\n" + "="*70 + "\n")

        env.close()

    except Exception as e:
        print(f"\n❌ Error loading environment '{env_name}': {e}\n")


def run_random_episode(env_name, num_steps=100, render=True, verbose=True):
    """Run one episode with random actions"""
    try:
        # Create environment
        render_mode = 'human' if render else None
        try:
            env = gym.make(env_name, render_mode=render_mode)
        except:
            env = gym.make(env_name)

        observation, info = env.reset(seed=42)

        if verbose:
            print("\n" + "="*70)
            print(f"RANDOM EPISODE: {env_name}")
            print("="*70)
            print(f"Initial observation: {observation}")
            print(f"Initial info: {info}")
            print("\nRunning episode with random actions...")
            print("-"*70)

        total_reward = 0
        step = 0

        for step in range(num_steps):
            # Random action
            action = env.action_space.sample()

            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if verbose and step % 10 == 0:
                print(f"Step {step:3}: Action={action}, "
                      f"Reward={reward:6.2f}, "
                      f"Total={total_reward:8.2f}")

            if render:
                time.sleep(0.02)  # Slow down for viewing

            if terminated or truncated:
                if verbose:
                    print(f"\nEpisode ended at step {step+1}")
                    print(f"Terminated: {terminated}, Truncated: {truncated}")
                break

        if verbose:
            print("-"*70)
            print(f"Total reward: {total_reward:.2f}")
            print(f"Total steps: {step+1}")
            print("="*70 + "\n")

        env.close()
        return total_reward, step+1

    except Exception as e:
        print(f"\n❌ Error running episode: {e}\n")
        return None, None


def interactive_mode(env_name):
    """
    Interactive mode where you can step through manually
    (Only works for discrete action spaces)
    """
    try:
        env = gym.make(env_name, render_mode='human')

        # Check if discrete action space
        if not hasattr(env.action_space, 'n'):
            print("❌ Interactive mode only works with discrete action spaces")
            env.close()
            return

        print("\n" + "="*70)
        print(f"INTERACTIVE MODE: {env_name}")
        print("="*70)
        print(f"\nAvailable actions: 0 to {env.action_space.n - 1}")
        print("Commands: [action_number], 'r' to reset, 'q' to quit\n")

        observation, info = env.reset()
        print(f"Initial observation: {observation}\n")

        total_reward = 0
        step = 0

        while True:
            user_input = input(f"Step {step} > ").strip().lower()

            if user_input == 'q':
                print("Quitting...")
                break

            if user_input == 'r':
                observation, info = env.reset()
                total_reward = 0
                step = 0
                print(f"Reset! New observation: {observation}\n")
                continue

            try:
                action = int(user_input)
                if action < 0 or action >= env.action_space.n:
                    print(f"Invalid action! Choose 0 to {env.action_space.n - 1}")
                    continue

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                print(f"  Observation: {observation}")
                print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")

                if terminated or truncated:
                    print(f"\n{'='*50}")
                    print(f"Episode ended! Total reward: {total_reward:.2f}")
                    print(f"{'='*50}\n")
                    observation, info = env.reset()
                    total_reward = 0
                    step = 0
                    print(f"Auto-reset! New observation: {observation}\n")

            except ValueError:
                print("Invalid input! Enter a number, 'r' to reset, or 'q' to quit")

        env.close()

    except Exception as e:
        print(f"\n❌ Error in interactive mode: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Explore Gymnasium environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gym-explorer.py list                    # List all environments
  python gym-explorer.py info CartPole-v1        # Show environment details
  python gym-explorer.py run CartPole-v1         # Run random episode
  python gym-explorer.py interactive CartPole-v1 # Manual control

Popular environments to try:
  - CartPole-v1          (balance a pole)
  - MountainCar-v0       (drive up a hill)
  - LunarLander-v3       (land a rocket)
  - Acrobot-v1           (swing up task)
  - FrozenLake-v1        (grid world)
        """
    )

    parser.add_argument(
        'command',
        choices=['list', 'info', 'run', 'interactive'],
        help='Command to execute'
    )

    parser.add_argument(
        'env_name',
        nargs='?',
        help='Environment name (required for info, run, interactive)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of steps for run command (default: 100)'
    )

    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering (faster)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    if args.command == 'list':
        list_environments()

    elif args.command == 'info':
        if not args.env_name:
            print("❌ Error: env_name required for 'info' command")
            parser.print_help()
            return
        show_env_info(args.env_name)

    elif args.command == 'run':
        if not args.env_name:
            print("❌ Error: env_name required for 'run' command")
            parser.print_help()
            return
        run_random_episode(
            args.env_name,
            num_steps=args.steps,
            render=not args.no_render,
            verbose=not args.quiet
        )

    elif args.command == 'interactive':
        if not args.env_name:
            print("❌ Error: env_name required for 'interactive' command")
            parser.print_help()
            return
        interactive_mode(args.env_name)


if __name__ == '__main__':
    main()

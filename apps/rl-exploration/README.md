# Gymnasium Environment Demos

Simple, standalone Python scripts to demonstrate RL environments without notebook pygame issues.

## 🎯 Why These Apps?

The Jupyter notebook `rl-gym-environments.ipynb` crashes due to pygame rendering conflicts. These standalone Python apps solve that problem by running directly from the terminal with proper pygame/OpenGL support.

## 📁 Available Demos

### 1. **mountaincar-demo.py** - MountainCar Strategies

Demonstrates different approaches to solving the MountainCar environment.

**Available Strategies:**
- `random`: Random actions (baseline, rarely succeeds)
- `right`: Always push right (fails - no momentum)
- `momentum`: Go left first to build momentum, then right (succeeds!)
- `heuristic`: Velocity-based decision making

**Usage:**
```bash
# Show environment info
python apps/rl-exploration/mountaincar-demo.py info

# Try the momentum strategy (builds momentum by going left first)
python apps/rl-exploration/mountaincar-demo.py momentum

# Try other strategies
python apps/rl-exploration/mountaincar-demo.py random
python apps/rl-exploration/mountaincar-demo.py right
python apps/rl-exploration/mountaincar-demo.py heuristic

# Compare all strategies (no rendering, shows stats)
python apps/rl-exploration/mountaincar-demo.py compare

# Run without rendering (faster)
python apps/rl-exploration/mountaincar-demo.py momentum --no-render

# Custom max steps
python apps/rl-exploration/mountaincar-demo.py momentum --steps 300
```

**Teaching Points:**
- Shows why simple strategies fail (always right has no momentum)
- Demonstrates importance of exploration (going "wrong" direction first)
- Perfect for explaining reward shaping and sparse rewards
- MountainCar is harder than it looks - great motivation for RL!

---

### 2. **cartpole-simple-demo.py** - CartPole Heuristics

Compare simple rule-based strategies vs baseline random agent.

**Available Strategies:**
- `random`: Random actions (~20-30 steps)
- `angle`: React to pole angle (~40-80 steps)
- `velocity`: React to angular velocity (~80-120 steps)
- `combined`: Best simple heuristic (~100-200 steps)

**Usage:**
```bash
# Show environment info
python apps/rl-exploration/cartpole-simple-demo.py info

# Try different strategies
python apps/rl-exploration/cartpole-simple-demo.py random
python apps/rl-exploration/cartpole-simple-demo.py angle
python apps/rl-exploration/cartpole-simple-demo.py velocity
python apps/rl-exploration/cartpole-simple-demo.py combined

# Compare all strategies (10 trials each)
python apps/rl-exploration/cartpole-simple-demo.py compare

# More trials for better statistics
python apps/rl-exploration/cartpole-simple-demo.py compare --trials 20

# Run without rendering
python apps/rl-exploration/cartpole-simple-demo.py combined --no-render
```

**Teaching Points:**
- Simple heuristics work somewhat (100-200 steps)
- But DQN achieves near-perfect 400-500 steps!
- Great motivation for why we need learning algorithms
- Shows the difference between hand-crafted rules and learned policies

---

### 3. **qlearning-mountaincar.py** - Tabular Q-Learning

Train a Q-learning agent on MountainCar with discretized state space.

**Commands:**
- `train`: Train a new Q-learning agent
- `test`: Test a trained agent
- `plot-training`: Visualize training progress
- `plot-qtable`: Visualize learned Q-values as heatmaps
- `info`: Show Q-table information

**Usage:**
```bash
# Train a Q-learning agent (saves qtable_mountaincar.pkl)
python apps/rl-exploration/qlearning-mountaincar.py train --episodes 5000

# Test the trained agent (with visualization)
python apps/rl-exploration/qlearning-mountaincar.py test

# Test without rendering (faster)
python apps/rl-exploration/qlearning-mountaincar.py test --no-render --test-episodes 10

# Visualize training progress
python apps/rl-exploration/qlearning-mountaincar.py plot-training

# Visualize learned Q-table
python apps/rl-exploration/qlearning-mountaincar.py plot-qtable

# Save plots to files
python apps/rl-exploration/qlearning-mountaincar.py plot-training --save-fig training.png
python apps/rl-exploration/qlearning-mountaincar.py plot-qtable --save-fig qtable.png

# Show Q-table info
python apps/rl-exploration/qlearning-mountaincar.py info
```

**Teaching Points:**
- State discretization (continuous → discrete)
- Q-table structure and initialization
- Q-learning update rule in action
- Exploration vs exploitation (epsilon-greedy)
- Convergence and learning curves
- Great bridge between heuristics and Deep Q-Networks

---

### 4. **qlearning_demo_app.py** - Interactive Q-Learning Tutorial (Streamlit)

An educational, blog-post style interactive demonstration of Q-learning.

**Features:**
- Step-by-step explanations with code snippets
- Live training with progress visualization
- Interactive hyperparameter tuning
- Q-table heatmap visualization
- Trajectory analysis of trained agent

**Usage:**
```bash
# Launch the interactive tutorial
streamlit run apps/rl-exploration/qlearning_demo_app.py

# Then open your browser to http://localhost:8501
```

**What You'll Learn:**
1. The MountainCar problem
2. State space discretization
3. Q-table initialization
4. Q-learning algorithm step-by-step
5. Train your own agent with custom hyperparameters
6. Visualize what the agent learned
7. Test the trained agent

**Teaching Points:**
- Perfect for classroom demonstrations
- Shows code alongside explanations
- Interactive exploration of hyperparameters
- Visual learning progress
- Immediate feedback on parameter choices
- Great for understanding before DQN

---

### 5. **gym-explorer.py** - Gymnasium Environment Browser

Explore all available Gymnasium environments, inspect action/observation spaces.

**Commands:**
- `list`: Show all available environments
- `info ENV_NAME`: Display environment details
- `run ENV_NAME`: Run random episode with visualization
- `interactive ENV_NAME`: Manual step-by-step control (discrete actions only)

**Usage:**
```bash
# List all available environments
python apps/rl-exploration/gym-explorer.py list

# Show details for a specific environment
python apps/rl-exploration/gym-explorer.py info CartPole-v1
python apps/rl-exploration/gym-explorer.py info MountainCar-v0
python apps/rl-exploration/gym-explorer.py info LunarLander-v3
python apps/rl-exploration/gym-explorer.py info Acrobot-v1

# Run random episode (watch random agent fail)
python apps/rl-exploration/gym-explorer.py run CartPole-v1
python apps/rl-exploration/gym-explorer.py run MountainCar-v0 --steps 200

# Interactive mode - control manually!
python apps/rl-exploration/gym-explorer.py interactive CartPole-v1
# Then type action numbers (0 or 1 for CartPole)
# Type 'r' to reset, 'q' to quit

# Run without rendering (faster, shows stats only)
python apps/rl-exploration/gym-explorer.py run CartPole-v1 --no-render --quiet
```

**Teaching Points:**
- Understand observation and action spaces
- See how different environments work
- Great for exploring before implementing RL algorithms
- Interactive mode helps build intuition

---

## 🎮 Complete Demo Workflow

Here's a suggested teaching sequence:

### 1. **Explore Environments** (5 minutes)
```bash
# See what's available
python apps/rl-exploration/gym-explorer.py list

# Understand CartPole
python apps/rl-exploration/gym-explorer.py info CartPole-v1

# Watch random agent fail
python apps/rl-exploration/gym-explorer.py run CartPole-v1
```

### 2. **Try Simple Heuristics** (10 minutes)
```bash
# CartPole: Compare strategies
python apps/rl-exploration/cartpole-simple-demo.py compare

# Show best heuristic
python apps/rl-exploration/cartpole-simple-demo.py combined

# Point: "Even our best heuristic only gets ~150, but DQN gets 500!"
```

### 3. **MountainCar Challenge** (10 minutes)
```bash
# Show the problem
python apps/rl-exploration/gym-explorer.py info MountainCar-v0

# Watch random agent fail
python apps/rl-exploration/mountaincar-demo.py random

# Watch "always right" fail
python apps/rl-exploration/mountaincar-demo.py right

# Show successful strategy
python apps/rl-exploration/mountaincar-demo.py momentum

# Compare all
python apps/rl-exploration/mountaincar-demo.py compare

# Point: "You need to go 'backwards' first - hard to discover!"
```

### 4. **Q-Learning Training** (30 minutes)
```bash
# Now learn Q-learning (tabular method)
python apps/rl-exploration/qlearning-mountaincar.py train --episodes 2000

# Test the trained agent
python apps/rl-exploration/qlearning-mountaincar.py test

# Visualize learning
python apps/rl-exploration/qlearning-mountaincar.py plot-training
python apps/rl-exploration/qlearning-mountaincar.py plot-qtable

# OR use interactive Streamlit demo
streamlit run apps/rl-exploration/qlearning_demo_app.py

# Point: "Q-learning works but limited to small state spaces!"
```

### 5. **DQN Training** (30-60 minutes)
```bash
# Now show them the power of Deep RL!
cd apps/rl-demo
python train_agent.py          # Train DQN on CartPole
python generate_videos.py      # Create videos
streamlit run demo_app.py      # Interactive dashboard

# Point: "DQN uses neural networks to handle large/continuous state spaces!"
```

### 6. **FlappyBird** (optional, 20 minutes)
```bash
cd apps/flappy-rl-demo
python play_game.py            # Human play
python play_with_agent.py      # Watch trained agent
streamlit run demo_app.py      # Full dashboard
```

---

## 🔧 Installation

All demos require:
```bash
pip install gymnasium[classic-control]
```

For Box2D environments (LunarLander, BipedalWalker):
```bash
pip install gymnasium[box2d]
```

For MuJoCo environments (optional, requires license):
```bash
pip install gymnasium[mujoco]
```

---

## 🐛 Troubleshooting

### "Kernel crashed" in Jupyter
**Solution:** Don't use Jupyter! Use these standalone Python scripts instead.

### "No module named gymnasium"
```bash
pip install gymnasium[classic-control]
```

### "Could not initialize video system"
- Make sure you're running from terminal, not Jupyter
- On macOS, the apps set `SDL_VIDEODRIVER=cocoa` automatically
- Try running without rendering: `--no-render`

### Window opens but freezes
- This is normal for slow rendering
- Try fewer steps: `--steps 50`
- Or disable rendering: `--no-render`

### "Environment not found"
```bash
# Check environment name spelling
python apps/rl-exploration/gym-explorer.py list

# Some environments need extra packages
pip install gymnasium[box2d]  # for LunarLander
```

---

## 📚 Environment Reference

### Classic Control (No extra install needed)
- **CartPole-v1**: Balance a pole on a cart (discrete actions, easy)
- **MountainCar-v0**: Drive up a hill (discrete actions, hard - sparse reward)
- **Acrobot-v1**: Swing up a two-link robot (discrete actions, medium)
- **Pendulum-v1**: Swing up a pendulum (continuous actions, medium)

### Box2D (Install: `pip install gymnasium[box2d]`)
- **LunarLander-v3**: Land a rocket (discrete, medium difficulty)
- **BipedalWalker-v3**: Walk with a 2D robot (continuous, hard)
- **CarRacing-v3**: Drive a car from pixels (continuous, hard)

### Toy Text (No extra install needed)
- **FrozenLake-v1**: Navigate a grid (discrete, simple)
- **Taxi-v3**: Pick up and drop off passengers (discrete, simple)
- **Blackjack-v1**: Play blackjack (discrete, simple)

---

## 🎓 Teaching Tips

### For Beginners
1. Start with `gym-explorer.py list` to see options
2. Use `gym-explorer.py info` to understand observation/action spaces
3. Watch random agents fail with `gym-explorer.py run`
4. Try simple heuristics with the demo scripts
5. Show how RL (DQN) outperforms heuristics

### For Intermediate Students
1. Compare heuristic strategies (`compare` mode)
2. Analyze why simple strategies fail
3. Discuss reward shaping and exploration
4. Run training with `rl-demo/train_agent.py`
5. Visualize learning with `rl-demo/demo_app.py`

### For Advanced Students
- Modify heuristics in the `.py` files
- Implement new strategies
- Try different environments
- Experiment with RL hyperparameters
- Implement improvements (Double DQN, Dueling DQN, etc.)

---

## 🔗 Related Demos

- **[rl-demo/](rl-demo/)**: Full DQN training on CartPole with checkpoints and videos
- **[flappy-rl-demo/](flappy-rl-demo/)**: DQN on FlappyBird with interactive gameplay
- **Notebook**: `notebooks/rl-gym-environments.ipynb` (reference only, use these apps instead)

---

## 📖 Further Reading

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Sutton & Barto - RL Book](http://incompleteideas.net/book/the-book-2nd.html)

---

## 💡 Quick Command Reference

```bash
# Environment exploration
python apps/rl-exploration/gym-explorer.py list
python apps/rl-exploration/gym-explorer.py info CartPole-v1
python apps/rl-exploration/gym-explorer.py run CartPole-v1

# CartPole demos (simple heuristics)
python apps/rl-exploration/cartpole-simple-demo.py info
python apps/rl-exploration/cartpole-simple-demo.py compare
python apps/rl-exploration/cartpole-simple-demo.py combined

# MountainCar demos (strategies)
python apps/rl-exploration/mountaincar-demo.py info
python apps/rl-exploration/mountaincar-demo.py compare
python apps/rl-exploration/mountaincar-demo.py momentum

# Q-Learning (tabular)
python apps/rl-exploration/qlearning-mountaincar.py train
python apps/rl-exploration/qlearning-mountaincar.py test
python apps/rl-exploration/qlearning-mountaincar.py plot-training
streamlit run apps/rl-exploration/qlearning_demo_app.py

# DQN training (deep RL - CartPole)
cd apps/rl-demo && python train_agent.py
cd apps/rl-demo && streamlit run demo_app.py

# FlappyBird (deep RL)
cd apps/flappy-rl-demo && python play_game.py
```

---

**Created to solve pygame/Jupyter notebook conflicts in RL teaching materials.**

# Platform-Jumping-Game
Platform Jumping Game base on Q-Learning

YouTube Link: https://youtu.be/gkcJrJq_hqU
# Q-Learning Platformer

# Q-Learning Platformer

This repository contains a platform-jumping game built with the Pygame library and trained using a Q-Learning algorithm. The agent learns to navigate through platforms, avoid enemies, and reach higher levels by interacting with the environment. The project includes separate modes for training and testing the agent and provides logs and plots for detailed performance analysis.

# Game Screen
<img width="151" alt="image" src="https://github.com/user-attachments/assets/40c474df-c907-4f85-80b8-2b9f1dda267e" />


## Overview

The project consists of two primary components:

- **Training Mode (`train.py`)**
  - Uses Q-Learning with an ε–greedy strategy to train an agent in a platformer environment.
  - Implements experience replay to stabilize the learning process.
  - Logs each episode’s total reward and ε value to a log file.
  - Generates learning curves (total reward and ε decay) along with statistical metrics such as mean reward, standard deviation, and moving average reward.

- **Testing Mode (`test.py`)**
  - Loads a trained Q-table to evaluate the agent's performance with a lower ε value (more exploitation).
  - Logs test episodes to a separate log file.
  - Generates test curves that plot the total rewards and ε decay. The test plots also display statistical metrics like the mean reward, standard deviation, and moving average reward.

## Features

- **Q-Learning Algorithm**  
  Implements a basic Q-Learning strategy including:
  - ε–greedy action selection
  - Q-value updates
  - Experience replay memory

- **Game Simulation**  
  Simulates a platformer environment with:
  - Multiple platforms at different levels
  - Physics for gravity, vertical and diagonal jumps
  - Dynamic enemy movement for collision challenges

- **Logging and Visualization**  
  - Training logs are saved to `learning_log.txt` and the trained Q-table is stored in `q_table.pkl`.
  - Test logs are saved to `test_log.txt`.
  - Reward curves (with mean and moving average) and ε decay curves are generated and saved as images.

## Project Structure

```
├── train.py             # Training script for Q-Learning agent
├── test.py              # Testing script for evaluating the trained agent
├── README.md            # Project documentation
```

By default, all logs, Q-tables, and plots are saved in the directory specified by the `save_dir` variable:
```
/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Assignment1/log/alpha0.1/
```
You can change this path by modifying the `save_dir` variable in both scripts.

## Setup and Installation

### Prerequisites

- Python 3.x
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

### Installation

You can install the required packages using pip:

```bash
pip install pygame numpy matplotlib pandas
```

## Usage

### Training Mode

To start the training process, run:

```bash
python train.py
```

During training:
- A game window will open, showing the agent interacting with platforms and enemies.
- The training log is appended to `learning_log.txt`.
- After each episode, the Q-table is updated and saved to `q_table.pkl`.
- Learning curves (total reward curve and ε decay curve) are generated, showing statistical metrics such as the mean reward, standard deviation, and moving average reward.

### Testing Mode

After training, you can test the agent by running:

```bash
python test.py
```

In testing mode:
- The trained Q-table is loaded from `q_table.pkl`.
- The agent is evaluated using a lower ε value (e.g., 0.05) to favor exploitation.
- Test logs are appended to `test_log.txt`.
- Test curves are generated, including the test total reward curve (with mean, standard deviation, and moving average) and the ε decay curve.

## Code Explanation

- **QAgent Class**:  
  This class implements the Q-Learning algorithm, featuring:
  - ε–greedy strategy for action selection.
  - Q-value update function using the standard Q-Learning update rule.
  - Experience replay to update the Q-table across multiple past experiences.

- **AgentObj Class**:  
  Represents the in-game agent (player) including its size, starting position, vertical speed, and health.

- **EnemyObj Class**:  
  Represents enemy characters that move left and right along the platforms. These serve as dynamic obstacles that the agent must avoid.

- **Game Environment Functions**:  
  Functions such as `draw_environment()`, `draw_UI()`, and `get_state()` are used to render the game scene, display the current status, and discretize the agent’s state, respectively.

- **Jump Simulation and Repositioning**:  
  Functions `simulate_jump()` and `reposition()` handle the physics of agent movement (jumping and horizontal repositioning) and manage collision detection with platforms and enemies.

- **Logging and Plotting**:  
  Both scripts include functions to log episode outcomes and plot learning curves with statistical metrics.

# Q-Learning Alpha Parameter Analysis

# Epsilon Decay Curve(showcase version -- alpha=0.2)
![train_epsilon_curve](https://github.com/user-attachments/assets/c0a69946-866a-4f32-b373-f98c837b4501)

# Total Reward Curve(showcase version -- alpha=0.2)
![train_learning_curve](https://github.com/user-attachments/assets/cb7cfa63-00bd-4101-b124-19c3f408111e)


This table summarizes the impact of different `alpha` (learning rate) values on training and testing performance.

| Alpha  | Training Stability | Training Reward | Testing Reward | Testing Stability | Recommended Scenario |
|--------|--------------------|----------------|----------------|--------------------|----------------------|
| **0.1** | Medium (Std Dev 51.83) | **Highest Training Reward (12.97)** | 13.68 (Low) | High Variance | Suitable for environments that require stable training, but testing performance may be suboptimal. |
| **0.2** | **Most Stable (Std Dev 40.06)** | 10.65 (Second Highest) | 15.50 (Medium) | **Most Stable (Std Dev 69.56)** | **Best trade-off: stable training and decent testing performance.** |
| **0.3** | High Training Variance (Std Dev 50.92) | **Lowest Training Reward (5.26)** | **Highest Testing Reward (35.57)** | High Variance | If the goal is the best final testing performance, `alpha=0.3` might be the optimal choice. |

# Training Mode Workflow Chart
![training_flowchart](https://github.com/user-attachments/assets/0a10eaeb-1727-4cba-a392-2e0b931b15e1)

# Testing Mode Workflow Chart
![testing_flowchart](https://github.com/user-attachments/assets/4425cdc3-8c40-4083-9ac5-2489c7614b98)


## Key Takeaways:
- **If you prioritize stable training**, `alpha=0.2` is the best choice.
- **If you want the highest testing performance**, `alpha=0.3` performs the best but has unstable training.
- **If you want a balanced approach**, `alpha=0.2` provides stable training and decent testing results.

**Recommendation**: Choose `alpha` based on whether you prioritize training stability or final performance.

## License

This project is released under the [MIT License](LICENSE).

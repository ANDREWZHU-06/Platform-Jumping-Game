# Platform-Jumping-Game
Platform Jumping Game base on Q-Learning
# Q-Learning Platformer

# Q-Learning Platformer

This repository contains a platform-jumping game built with the Pygame library and trained using a Q-Learning algorithm. The agent learns to navigate through platforms, avoid enemies, and reach higher levels by interacting with the environment. The project includes separate modes for training and testing the agent and provides logs and plots for detailed performance analysis.

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

## License

This project is released under the [MIT License](LICENSE).

---

Feel free to fork this repository and modify the code for your research or educational purposes. If you have any questions or suggestions, please open an issue or submit a pull request.
```

---

This README provides a comprehensive explanation of the project, usage instructions, code structure, and features, all in English.

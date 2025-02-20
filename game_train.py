import pygame
import random
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# -----------------------------------------
# Specify the directory for saving Q-tables and log files.
# -----------------------------------------
save_dir = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Assignment1/log/alpha0.1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

q_table_path = os.path.join(save_dir, "q_table.pkl")
log_file_path = os.path.join(save_dir, "learning_log.txt")

# -------------------------------------------------------------------
# Function: save_q_table
#
# Saves the current Q-table (a Python dictionary) into a pickle file.
# This allows the training results to be stored and loaded later.
# -------------------------------------------------------------------
def save_q_table(q_table):
    """Save the Q-table to a pickle file."""
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table has been saved to: {q_table_path}")

# -------------------------------------------------------------------
# Function: load_q_table
#
# Loads the Q-table from the specified file. If the file does not exist,
# it returns an empty dictionary to start with.
# -------------------------------------------------------------------
def load_q_table():
    """Load the saved Q-table from a file if it exists; otherwise, return an empty dictionary."""
    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
        print(f"Q table has been loaded from {q_table_path}")
        return q_table
    else:
        print("No saved Q-table found. Initializing a new one.")
        return {}

# -------------------------------------------------------------------
# Function: log_episode
#
# This function writes the progress of an episode (its number, total reward,
# and current epsilon value) to the log file.
# The log is written as a plain text line.
# -------------------------------------------------------------------
def log_episode(episode, total_reward, epsilon):
    """
    Record the learning progress in the log file.
    Example format: "Episode: 1, Total Reward: -51.50, Epsilon: 1.000"
    """
    log_line = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}\n"
    with open(log_file_path, "a") as f:
        f.write(log_line)
    print("Learning progress：", log_line.strip())


# -------------------------
# Game initialization parameters
# -------------------------
WINDOW_WIDTH = 400 # Width of the game window.
WINDOW_HEIGHT = 600 # Height of the game window.
FPS = 60 # Frames per second.

# Physics parameters for the game:
GRAVITY = 0.5
JUMP_VELOCITY = -12  # Initial jump velocity (vertical component)
HORIZONTAL_VELOCITY = 5  # Horizontal component when jumping diagonally
MOVE_STEP = 10  # Number of pixels the agent moves horizontally per action.

# -------------------------------------------------------------------
# List of platforms:
# Each platform is defined by its "level" and its rectangle (pygame.Rect).
# These platforms act as ground and elevated surfaces in the game.
# -------------------------------------------------------------------
platforms = [
    {"level": 0, "rect": pygame.Rect(0, 550, 400, 50)},  # ground
    {"level": 1, "rect": pygame.Rect(20, 450, 150, 10)},  # Level 1 - left platform
    {"level": 1, "rect": pygame.Rect(230, 450, 150, 10)},  # Level 1 - right platform
    {"level": 2, "rect": pygame.Rect(30, 350, 100, 10)},  # Level 2 - left platform
    {"level": 2, "rect": pygame.Rect(270, 350, 100, 10)},  # Level 2 - right platform
    {"level": 3, "rect": pygame.Rect(50, 250, 80, 10)},  # Level 3 - left platform
    {"level": 3, "rect": pygame.Rect(270, 250, 80, 10)},  # Level 3 - right platform
    {"level": 4, "rect": pygame.Rect(150, 150, 100, 10)},  # Level 4 - top platform
]


# -------------------------------------------------------------------
# QAgent Class:
# Implements the Q-Learning algorithm.
# This class includes methods to select actions using an epsilon-greedy
# strategy, update Q-values, and store past experiences.
# -------------------------------------------------------------------
class QAgent:
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate (set to 0.1/0.2/0.3)
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Q-table: key is state, value is a list of 5 Q-values corresponding to 5 actions.
        self.experience_replay = []  # Experience replay memory storage
        self.MAX_MEMORY = 5000  # Maximum number of experiences stored

    def get_q_values(self, state):
        """
        Get the Q-values for the current state.
        If the state is not present in the Q-table, initialize it with 5 actions having Q-value 0.0.
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(5)]
        return self.q_table[state]

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
          - With probability epsilon, choose a random action (exploration).
          - Otherwise, choose the action with the highest Q-value (exploitation).
        """
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        else:
            q_vals = self.get_q_values(state)
            max_q = max(q_vals)
            best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
            return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        """
        Update the Q-value for the state-action pair using the formula:
            Q(s, a) = Q(s, a) + alpha * (reward + gamma * max[Q(next_state)] - Q(s, a))
        """
        old_q = self.get_q_values(state)[action]
        next_max = max(self.get_q_values(next_state)) if next_state is not None else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q

    def store_experience(self, state, action, reward, next_state):
        """
        Store a tuple (state, action, reward, next_state) in the experience replay pool.
        Remove the oldest experience if the pool exceeds the maximum memory size.
        """
        if len(self.experience_replay) > self.MAX_MEMORY:
            self.experience_replay.pop(0)
        self.experience_replay.append((state, action, reward, next_state))

    def replay_experience(self, batch_size=64):
        """
        Randomly sample a batch of experiences from the replay memory and update Q-values for each.
        """
        if len(self.experience_replay) > batch_size:
            batch = random.sample(self.experience_replay, batch_size)
            for state, action, reward, next_state in batch:
                self.update_q(state, action, reward, next_state)


# -------------------------------------------------------------------
# AgentObj Class:
# Represents the game agent (player). This class initializes the agent's size,
# position, and health.
# In-game [agent] object controlled by Q-learning
# -------------------------
class AgentObj:
    def __init__(self):
        self.width = 20
        self.height = 20
        self.reset()

    def reset(self):
        """
        Reset the agent's position to a random location on the ground platform.
        Also resets vertical velocity and health.
        """
        ground = platforms[0]['rect']
        self.x = random.randint(ground.left, ground.right - self.width)
        self.y = ground.top - self.height
        self.vy = 0
        self.health = 100


# -------------------------------------------------------------------
# EnemyObj Class:
# Represents enemy characters that move horizontally along a platform.
# Enemies have a random starting position and direction, and reverse direction
# upon reaching the edge of their platform.
# -------------------------------------------------------------------
class EnemyObj:
    def __init__(self, platform):
        self.platform = platform
        pf_rect = platform['rect']
        self.width = 20
        self.height = 20
        self.x = random.randint(pf_rect.left, pf_rect.right - self.width)
        self.y = pf_rect.top - self.height
        self.speed = 2
        self.direction = random.choice([-1, 1])

    def update(self):
        """
        Update the enemy's horizontal position.
        Reverse direction if the enemy crashes into the platform boundaries.
        """
        self.x += self.speed * self.direction
        pf_rect = self.platform['rect']
        if self.x <= pf_rect.left or self.x + self.width >= pf_rect.right:
            self.direction *= -1

    def draw(self, screen):
        # Draw enemy as a circle with eyes and a simple mouth.
        center = (int(self.x + self.width / 2), int(self.y + self.height / 2))
        radius = self.width // 2
        pygame.draw.circle(screen, (148, 0, 211), center, radius)
        # Draw two eyes
        eye_radius = 2
        eye_offset_x = 4
        eye_offset_y = 4
        left_eye = (center[0] - eye_offset_x, center[1] - eye_offset_y)
        right_eye = (center[0] + eye_offset_x, center[1] - eye_offset_y)
        pygame.draw.circle(screen, (0, 0, 0), left_eye, eye_radius)
        pygame.draw.circle(screen, (0, 0, 0), right_eye, eye_radius)
        # Draw a mouth as a short line
        start_mouth = (center[0] - radius // 2, center[1] + 2)
        end_mouth = (center[0] + radius // 2, center[1] + 2)
        pygame.draw.line(screen, (0, 0, 0), start_mouth, end_mouth, 1)

    def get_rect(self):
        """
        Return a rectangle that represents the enemy's current position.
        This is used for collision detection.
        """
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)


# -------------------------------------------------------------------
# Drawing Functions:
# These functions handle drawing the agent, the game environment,
# and the user interface (information overlay).
# -------------------------------------------------------------------
def draw_agent(agent):
    """
    Draw the agent on the screen.
    The agent is depicted as a green circle with eyes and a mouth.
    """
    center = (int(agent.x + agent.width / 2), int(agent.y + agent.height / 2))
    radius = agent.width // 2
    pygame.draw.circle(screen, (0, 200, 0), center, radius)
    eye_radius = 2
    eye_offset_x = 4
    eye_offset_y = 4
    left_eye = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    right_eye = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    pygame.draw.circle(screen, (0, 0, 0), left_eye, eye_radius)
    pygame.draw.circle(screen, (0, 0, 0), right_eye, eye_radius)
    mouth_rect = pygame.Rect(center[0] - radius // 2, center[1], radius, radius // 2)
    pygame.draw.arc(screen, (0, 0, 0), mouth_rect, 3.14, 2 * 3.14, 1)


def draw_environment(agent):
    """
    Clear the screen and draw the background along with all platforms and the agent.
    """
    screen.fill((238, 245, 219))  # Background color
    for pf in platforms:
        pygame.draw.rect(screen, (20, 197, 139), pf['rect'])
    draw_agent(agent)


def draw_UI(episode, epsilon, health, curr_score, high_score):
    """
    Display the user interface overlay that shows:
    - The current episode number.
    - The current epsilon value.
    - The agent's health.
    - The current episode's score.
    - The highest score achieved so far.
    """
    ui_text1 = f"Episode: {episode}"
    ui_text2 = f"Epsilon: {epsilon:.3f}"
    ui_text3 = f"Health: {health}"
    ui_text4 = f"Score: {curr_score}"
    ui_text5 = f"High Score: {high_score}"
    texts = [ui_text1, ui_text2, ui_text3, ui_text4, ui_text5]
    y_offset = 5
    for text in texts:
        text_surface = font.render(text, True, (20, 17, 22))
        screen.blit(text_surface, (5, y_offset))
        y_offset += 25


# -------------------------------------------------------------------
# Function: get_state
#
# Discretizes the agent’s state based on its relative horizontal position
# on its current platform. The state is represented as a tuple: (platform level, bin index).
# The bin index (0-9) divides the platform into 10 equal sections.
# -------------------------------------------------------------------
def get_state(agent, current_platform):
    """
    Return a discrete state representation consisting of:
      - The platform level.
      - The bin index (0-9) of the agent's relative x-position within the platform.
    """
    p_rect = current_platform['rect']
    relative_x = agent.x - p_rect.left
    effective_width = p_rect.width - agent.width
    bin_index = int((relative_x / effective_width) * 10) if effective_width else 0
    bin_index = max(0, min(9, bin_index))
    return (current_platform['level'], bin_index)


# -------------------------------------------------------------------
# Function: handle_pause
# Pauses the game until the user presses the "P" key.
# A pause screen is displayed during this time.
# -------------------------------------------------------------------
def handle_pause():
    paused = True
    pause_text = font.render("PAUSED - Press P to resume", True, (255, 255, 0))
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = False
        screen.fill((0, 0, 0))
        screen.blit(pause_text, (50, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        clock.tick(FPS)


def show_start_screen():
    """
    Display the start screen with a title and instruction.
    Wait until the user presses any key to start the game.
    """
    screen.fill((0, 0, 0))
    title_font = pygame.font.SysFont(None, 48)
    msg_font = pygame.font.SysFont(None, 24)
    title_text = title_font.render("Q-Learning Platformer", True, (255, 255, 255))
    instruct_text = msg_font.render("Press any key to start", True, (255, 255, 0))
    screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, WINDOW_HEIGHT // 2 - 50))
    screen.blit(instruct_text, (WINDOW_WIDTH // 2 - instruct_text.get_width() // 2, WINDOW_HEIGHT // 2 + 10))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting = False
        clock.tick(FPS)


# -------------------------------------------------------------------
# Function: display_episode_end_message
#
# Displays an on-screen message at the end of an episode, showing the final score.
# -------------------------------------------------------------------
def display_episode_end_message(episode, score):
    message = f"Episode {episode} Over! Score: {score}"
    text_surface = font.render(message, True, (255, 0, 0))
    screen.fill((0, 0, 0))
    screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2,
                               WINDOW_HEIGHT // 2 - text_surface.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(1000)


# -------------------------
# Repositioning function for horizontal movements on the current platform.
# -------------------------
def reposition(agent, action, current_platform, enemies):
    """
    Update the agent's position according to a horizontal move action.
      - If action==0: move left.
      - If action==1: move right.
    Also update enemy positions and check for collisions.
    Returns:
      new_state: the new discrete state.
      reward: base reward with collision penalty.
      terminal: game over flag (if agent's health is 0 or below).
    """
    pf_rect = current_platform['rect']
    if action == 0:
        agent.x = max(pf_rect.left, agent.x - MOVE_STEP)
    elif action == 1:
        agent.x = min(pf_rect.right - agent.width, agent.x + MOVE_STEP)

    for enemy in enemies:
        enemy.update()

    agent_rect = pygame.Rect(int(agent.x), int(agent.y), agent.width, agent.height)
    collision_occurred = False
    for enemy in enemies:
        if agent_rect.colliderect(enemy.get_rect()):
            collision_occurred = True
            agent.health -= 20
    reward = -0.1
    if collision_occurred:
        reward += -20
    new_state = get_state(agent, current_platform)
    terminal = agent.health <= 0
    return new_state, reward, terminal


# -------------------------
# Jump simulation function.
# It simulates the physics of a jump (vertical/diagonal) and monitors for landing or terminal events.
# -------------------------
def simulate_jump(agent, action, current_platform, enemies):
    """
    Simulate a jump according to the selected action:
      - action 2: vertical jump (no horizontal movement)
      - action 3: jump diagonally to the left
      - action 4: jump diagonally to the right
    The simulation runs frame-by-frame until the agent lands on a platform or a terminal condition occurs.
    Returns:
      landing_platform: the platform the agent landed on (if any)
      new_state: the discrete state after landing
      reward: accumulated reward (including bonus for landing on higher platforms)
      terminal: game over flag.
    """
    if action == 2:
        vx, vy = 0, JUMP_VELOCITY
    elif action == 3:
        vx, vy = -HORIZONTAL_VELOCITY, JUMP_VELOCITY
    elif action == 4:
        vx, vy = HORIZONTAL_VELOCITY, JUMP_VELOCITY
    else:
        vx, vy = 0, JUMP_VELOCITY

    landed = False
    landing_platform = None
    terminal = False
    collision_occurred = False
    reward = 0

    # Jump simulation loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                handle_pause()

        agent.x += vx
        agent.y += vy
        vy += GRAVITY  # Update vertical velocity due to gravity

        for enemy in enemies:
            enemy.update()

        draw_environment(agent)
        for enemy in enemies:
            enemy.draw(screen)
        # Display UI (global variables episode, current_score, high_score and q_agent.epsilon retrieved)
        draw_UI(episode, q_agent.epsilon, agent.health, current_score, high_score)
        pygame.display.flip()
        clock.tick(FPS)

        agent_rect = pygame.Rect(int(agent.x), int(agent.y), agent.width, agent.height)
        for enemy in enemies:
            if agent_rect.colliderect(enemy.get_rect()):
                if not collision_occurred:
                    collision_occurred = True
                    agent.health -= 20
                    reward += -20

        # Check for terminal condition (agent's health)
        if agent.health <= 0:
            terminal = True
            reward += -50
            break

        # Check if agent lands on a platform (only consider when falling down: vy > 0)
        if vy > 0:
            for pf in platforms:
                pf_rect = pf["rect"]
                if agent_rect.colliderect(pf_rect):
                    # Ensure landing is on top of the platform
                    if (agent.y + agent.height - vy) <= pf_rect.top + 5:
                        landed = True
                        landing_platform = pf
                        agent.y = pf_rect.top - agent.height  # Align agent's bottom with platform's top
                        vy = 0
                        break
        if landed:
            break

        # If the agent falls below the window (misses landing), mark terminal.
        if agent.y > WINDOW_HEIGHT:
            terminal = True
            reward += -50
            break

    # Determine new state based on landing result.
    if terminal and not landed:
        new_state = get_state(agent, current_platform)
    else:
        new_state = get_state(agent, landing_platform)
        # Reward based on the change in platform level:
        if landing_platform["level"] > current_platform["level"]:
            level_diff = landing_platform["level"] - current_platform["level"]
            reward += 50 * level_diff
        elif landing_platform["level"] == current_platform["level"]:
            reward += -1
        else:
            reward += -30

    return landing_platform, new_state, reward, terminal


# -------------------------------------------------------------------
# Pygame initialization and window configuration for training mode.
# -------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Platformer - Train Mode")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# -------------------------------------------------------------------
# Instantiate the Q-learning agent and game objects.
# example: The agent starts with full exploration (epsilon = 1.0), a learning rate of 0.1,
# and a discount factor of 0.9.
# -------------------------------------------------------------------
q_agent = QAgent(epsilon=1.0, alpha=0.1, gamma=0.9)
player = AgentObj()
current_score = 0
high_score = 0
episode = 1


# -------------------------------------------------------------------
# Function: train
#
# Main training loop:
# - Resets the agent for each episode.
# - The agent interacts with the environment (moves, jumps, and collides).
# - Q-values are updated and experiences are stored.
# - Logs are written and the Q-table is saved after each episode.
# - Epsilon is decayed gradually to shift from exploration to exploitation.
# -------------------------------------------------------------------
def train():
    global episode, current_score, high_score, q_agent, player
    show_start_screen()
    while True:
        episode_reward = 0
        terminal = False

        player.reset() # Reset player's position and health.
        current_platform = platforms[0] # Start on the ground platform.

        # Create enemies on platforms with levels 1, 2, or 3.
        enemies = [EnemyObj(pf) for pf in platforms if pf["level"] in [1, 2, 3]]

        while not terminal:
            state = get_state(player, current_platform)
            action = q_agent.choose_action(state)

            # For actions 0 and 1, perform horizontal repositioning.
            if action in [0, 1]:
                new_state, reward, term = reposition(player, action, current_platform, enemies)
            else:
                # Actions 2, 3, 4 simulate jump actions.
                landing_platform, new_state, reward, term = simulate_jump(player, action, current_platform, enemies)
                if not term and landing_platform is not None:
                    current_platform = landing_platform

            terminal = term
            episode_reward += reward
            # Update Q-table with the experience
            q_agent.update_q(state, action, reward, new_state if new_state is not None else state)
            q_agent.store_experience(state, action, reward, new_state)

            pygame.time.wait(200) # Delay to control simulation speed.

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    handle_pause()

        # At end of each episode, perform experience replay update.
        q_agent.replay_experience()
        print(f"Episode: {episode}, Total Reward: {episode_reward:.2f}, Epsilon: {q_agent.epsilon:.3f}")
        current_score = episode_reward
        if episode_reward > high_score:
            high_score = episode_reward

        log_episode(episode, episode_reward, q_agent.epsilon)
        episode += 1
        # Decay epsilon gradually, but never lower than 0.1.
        q_agent.epsilon = max(0.1, q_agent.epsilon * 0.995)
        pygame.time.wait(1000)
        save_q_table(q_agent.q_table)


# -------------------------------------------------------------------
# Function: plot_learning_curve_and_save
#
# Reads the training log file to extract episode numbers, total rewards, and epsilon values.
# Plots two graphs: the total reward curve (with mean and moving average) and the epsilon decay curve.
# Statistical metrics (mean reward and standard deviation) are calculated and displayed.
# -------------------------------------------------------------------
MOVING_AVG_WINDOW = 50
def plot_learning_curve_and_save():
    """
    Read the learning log file, extract episodes, total rewards, and epsilon values,
    then plot and save both the total reward curve and the epsilon decay curve.
    Additionally, compute and display statistical metrics like mean reward, standard deviation,
    and moving average reward.
    """
    episodes = []
    total_rewards = []
    epsilons = []

    try:
        # log_file_path = "learning_log.txt"  # 确保你的日志文件路径正确
        # save_dir = "results"  # 结果保存目录
        # os.makedirs(save_dir, exist_ok=True)

        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析日志格式: "Episode: 10, Total Reward: -20.30, Epsilon: 0.956"
                m = re.search(r"Episode:\s*(\d+),\s*Total Reward:\s*([-+]?\d*\.?\d+),\s*Epsilon:\s*([-+]?\d*\.?\d+)", line)
                if m:
                    episodes.append(int(m.group(1)))
                    total_rewards.append(float(m.group(2)))
                    epsilons.append(float(m.group(3)))
    except Exception as e:
        print("Read log file error:", e)
        return

    if not episodes:
        print("No data found in the log file.")
        return

    # caculate moving average
    mean_reward = np.mean(total_rewards)  # Mean reward
    std_reward = np.std(total_rewards)  # Reward standard deviation
    moving_avg_rewards = pd.Series(total_rewards).rolling(window=MOVING_AVG_WINDOW, min_periods=1).mean()  # 移动平均奖励

    # Plot Total Reward Curve.
    plt.figure(figsize=(10, 5))

    # original reward curve
    plt.plot(episodes, total_rewards, marker='o', color='b', label='Total Reward', alpha=0.5)

    # mean reward curve (dashed line）
    plt.axhline(y=mean_reward, color='g', linestyle='--', label=f'Mean Reward ({mean_reward:.2f})')

    # moving average reward curve (solid line)
    plt.plot(episodes, moving_avg_rewards, color='r', linestyle='-', linewidth=2, label='Moving Avg Reward')

    # title and legend
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward Curve for Q-learning')
    plt.legend()
    plt.grid(True)

    # statical metrics
    stats_text = f"Mean Reward: {mean_reward:.2f}\nStd Dev: {std_reward:.2f}"
    plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # save & show figure
    plot_reward_path = os.path.join(save_dir, "train_learning_curve.png")
    plt.savefig(plot_reward_path)
    print(f"Total reward curve saved to: {plot_reward_path}")
    plt.close()

    # plot epsilon decay curve
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, marker='x', color='r', label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Curve for Q-learning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_epsilon_path = os.path.join(save_dir, "train_epsilon_curve.png")
    plt.savefig(plot_epsilon_path)
    print(f"Epsilon decay curve saved to: {plot_epsilon_path}")
    plt.close()

# -------------------------------------------------------------------
# Main entry point for training.
# Runs the training loop and then plots the learning curves once training is complete.
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        print("Training finished.")
        plot_learning_curve_and_save()

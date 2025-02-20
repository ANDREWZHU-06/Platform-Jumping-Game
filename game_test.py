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
# Specify the directory  (Q-tables and logs)
# -----------------------------------------
save_dir = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Assignment1/log/alpha0.1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

q_table_path = os.path.join(save_dir, "q_table.pkl")
log_file_path = os.path.join(save_dir, "learning_log.txt")
test_log_path = os.path.join(save_dir, "test_log.txt")

def save_q_table(q_table):
    """Save the Q-table to a file"""
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table has been saved to: {q_table_path}")

def load_q_table():
    """try to load the Q-table from a file, return an empty table if not found"""
    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
        print(f"Q table has been loaded from {q_table_path}")
        return q_table
    else:
        print("No saved Q-table found. Initializing a new one.")
        return {}

def log_episode(episode, total_reward, epsilon):
    """
    Record the status of each Episode to a log (which can be used to observe the effect during testing)
    """
    log_line = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}\n"
    with open(log_file_path, "a") as f:
        f.write(log_line)
    print("log has been recorded：", log_line.strip())

def log_test_episode(episode, total_reward, epsilon):
    """
    Add a new test process logging function, specifically for logging the test mode, will not overwrite learning_log.txt
    """
    log_line = f"Test Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}\n"
    with open(test_log_path, "a") as f:
        f.write(log_line)
    print("Test Log:", log_line.strip())

# -------------------------
# Game initialization
# -------------------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
FPS = 60

GRAVITY = 0.5
JUMP_VELOCITY = -12         # jump velocity
HORIZONTAL_VELOCITY = 5      # jump horizontal velocity
MOVE_STEP = 10               # move step

# -------------------------
# platforms settings
# -------------------------
platforms = [
    {"level": 0, "rect": pygame.Rect(0, 550, 400, 50)},
    {"level": 1, "rect": pygame.Rect(20, 450, 150, 10)},
    {"level": 1, "rect": pygame.Rect(230, 450, 150, 10)},
    {"level": 2, "rect": pygame.Rect(30, 350, 100, 10)},
    {"level": 2, "rect": pygame.Rect(270, 350, 100, 10)},
    {"level": 3, "rect": pygame.Rect(50, 250, 80, 10)},
    {"level": 3, "rect": pygame.Rect(270, 250, 80, 10)},
    {"level": 4, "rect": pygame.Rect(150, 150, 100, 10)},
]

# -------------------------
# Q-learning algorithm settings(testing mode)
# -------------------------
class QAgent:
    def __init__(self, epsilon=0.05, alpha=0.3, gamma=0.9):
        self.epsilon = epsilon  # setting for epsilon-greedy
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}  # load_q_table() from training

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(5)]
        return self.q_table[state]

    def choose_action(self, state):
        q_vals = self.get_q_values(state)
        max_q = max(q_vals)
        best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
        # here we use epsilon-greedy to choose the action, but mainly we use the best action
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        else:
            return random.choice(best_actions)

# -------------------------
# In-game Agent objects (controlled by Q-learning)
# -------------------------
class AgentObj:
    def __init__(self):
        self.width = 20
        self.height = 20
        self.reset()
    def reset(self):
        ground = platforms[0]['rect']
        self.x = random.randint(ground.left, ground.right - self.width)
        self.y = ground.top - self.height
        self.vy = 0
        self.health = 100

# -------------------------
# Enemy (NPC) objects: Move left and right
# -------------------------
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
        self.x += self.speed * self.direction
        pf_rect = self.platform['rect']
        if self.x <= pf_rect.left or self.x + self.width >= pf_rect.right:
            self.direction *= -1
    def draw(self, screen):
        center = (int(self.x + self.width/2), int(self.y + self.height/2))
        radius = self.width // 2
        pygame.draw.circle(screen, (148, 0, 211), center, radius)
        eye_radius = 2
        eye_offset_x = 4
        eye_offset_y = 4
        left_eye = (center[0] - eye_offset_x, center[1] - eye_offset_y)
        right_eye = (center[0] + eye_offset_x, center[1] - eye_offset_y)
        pygame.draw.circle(screen, (0, 0, 0), left_eye, eye_radius)
        pygame.draw.circle(screen, (0, 0, 0), right_eye, eye_radius)
        start_mouth = (center[0] - radius // 2, center[1] + 2)
        end_mouth = (center[0] + radius // 2, center[1] + 2)
        pygame.draw.line(screen, (0, 0, 0), start_mouth, end_mouth, 1)
    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

# -------------------------
# Functions for drawing agent and environments
# -------------------------
def draw_agent(agent):
    center = (int(agent.x + agent.width/2), int(agent.y + agent.height/2))
    radius = agent.width // 2
    pygame.draw.circle(screen, (0, 200, 0), center, radius)
    eye_radius = 2
    eye_offset_x = 4
    eye_offset_y = 4
    left_eye = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    right_eye = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    pygame.draw.circle(screen, (0, 0, 0), left_eye, eye_radius)
    pygame.draw.circle(screen, (0, 0, 0), right_eye, eye_radius)
    mouth_rect = pygame.Rect(center[0] - radius//2, center[1], radius, radius//2)
    pygame.draw.arc(screen, (0, 0, 0), mouth_rect, 3.14, 2*3.14, 1)

def draw_environment(agent):
    screen.fill((238, 245, 219))
    for pf in platforms:
        pygame.draw.rect(screen, (20, 197, 139), pf['rect'])
    draw_agent(agent)

def draw_UI(episode, epsilon, health, curr_score, high_score):
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

# -------------------------
# State discretisation: dividing the relative position of the agent on the platform into 10 intervals
# -------------------------
def get_state(agent, current_platform):
    p_rect = current_platform['rect']
    relative_x = agent.x - p_rect.left
    effective_width = p_rect.width - agent.width
    bin_index = int((relative_x / effective_width) * 10) if effective_width else 0
    bin_index = max(0, min(9, bin_index))
    return (current_platform['level'], bin_index)

def handle_pause():
    paused = True
    pause_text = font.render("PAUSED - Press P to resume", True, (255, 255, 0))
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = False
        screen.fill((0, 0, 0))
        screen.blit(pause_text, (50, WINDOW_HEIGHT//2))
        pygame.display.flip()
        clock.tick(FPS)

def show_start_screen():
    screen.fill((0, 0, 0))
    title_font = pygame.font.SysFont(None, 48)
    msg_font = pygame.font.SysFont(None, 24)
    title_text = title_font.render("Q-Learning Platformer - Test Mode", True, (255, 255, 255))
    instruct_text = msg_font.render("Press any key to start testing", True, (255, 255, 0))
    screen.blit(title_text, (WINDOW_WIDTH//2 - title_text.get_width()//2, WINDOW_HEIGHT//2 - 50))
    screen.blit(instruct_text, (WINDOW_WIDTH//2 - instruct_text.get_width()//2, WINDOW_HEIGHT//2 + 10))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting = False
        clock.tick(FPS)

def display_episode_end_message(episode, score):
    message = f"Episode {episode} Over! Score: {score}"
    text_surface = font.render(message, True, (255, 0, 0))
    screen.fill((0, 0, 0))
    screen.blit(text_surface, (WINDOW_WIDTH//2 - text_surface.get_width()//2,
                                 WINDOW_HEIGHT//2 - text_surface.get_height()//2))
    pygame.display.flip()
    pygame.time.wait(1000)

# -------------------------
# reposition the agent and enemies according to the action taken and the current platform
# -------------------------
def reposition(agent, action, current_platform, enemies):
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

def simulate_jump(agent, action, current_platform, enemies):
    # set the velocity and direction of the agent based on the action
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

    # record the previous y position of the agent
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                handle_pause()

        prev_y = agent.y             # record the previous y position of the agent
        prev_bottom = prev_y + agent.height

        agent.x += vx
        agent.y += vy
        vy += GRAVITY

        for enemy in enemies:
            enemy.update()

        draw_environment(agent)
        for enemy in enemies:
            enemy.draw(screen)
        draw_UI(episode, q_agent.epsilon, agent.health, current_score, high_score)
        pygame.display.flip()
        clock.tick(FPS)

        agent_rect = pygame.Rect(int(agent.x), int(agent.y), agent.width, agent.height)
        # detect collision with enemies
        for enemy in enemies:
            if agent_rect.colliderect(enemy.get_rect()):
                if not collision_occurred:
                    collision_occurred = True
                    agent.health -= 20
                    reward += -20

        if agent.health <= 0:
            terminal = True
            reward += -50
            break

        # detect collision with platforms:
        # 1. check if the agent is landed on a platform
        # 2. if the agent is landed, adjust its position to the top of the platform
        # 3. if the agent is not landed, check if it is colliding with any platform
        # 4. if the agent is colliding with a platform, adjust its position to the top of the platform
        # 5. if the agent is not colliding with any platform, check if it is out of the screen
        if vy > 0:
            curr_bottom = agent.y + agent.height
            for pf in platforms:
                pf_rect = pf["rect"]

                if prev_bottom <= pf_rect.top + 5 and curr_bottom >= pf_rect.top:
                    if agent_rect.colliderect(pf_rect):
                        landed = True
                        landing_platform = pf

                        agent.y = pf_rect.top - agent.height
                        vy = 0
                        break
            if landed:
                break

        if agent.y > WINDOW_HEIGHT:
            terminal = True
            reward += -50
            break

    if terminal and not landed:
        new_state = get_state(agent, current_platform)
    else:
        new_state = get_state(agent, landing_platform)
        if landing_platform["level"] > current_platform["level"]:
            level_diff = landing_platform["level"] - current_platform["level"]
            reward += 50 * level_diff
        elif landing_platform["level"] == current_platform["level"]:
            reward += -1
        else:
            reward += -30

    return landing_platform, new_state, reward, terminal

# -------------------------
# Pygame initialization and main loop
# -------------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Platformer - Test Mode")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# create the objects for the game and load the trained q-table
q_agent = QAgent(epsilon=0.05, alpha=0.3, gamma=0.9)
q_agent.q_table = load_q_table()  # load the trained q-table
player = AgentObj()
current_score = 0
high_score = 0
episode = 1

# -------------------------
# test main loop
# -------------------------
def test():
    global episode, current_score, high_score, q_agent, player
    show_start_screen()

    while True:
        episode_reward = 0
        terminal = False

        player.reset()
        current_platform = platforms[0]

        # generate the enemies for the current platform (level 1-3）
        enemies = [EnemyObj(pf) for pf in platforms if pf["level"] in [1, 2, 3]]

        while not terminal:
            state = get_state(player, current_platform)
            action = q_agent.choose_action(state)
            if action in [0, 1]:
                new_state, reward, term = reposition(player, action, current_platform, enemies)
                terminal = term
                episode_reward += reward
            else:
                landing_platform, new_state, reward, term = simulate_jump(player, action, current_platform, enemies)
                terminal = term
                episode_reward += reward
                if not terminal and landing_platform is not None:
                    current_platform = landing_platform

            pygame.time.wait(200)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    handle_pause()

        print(f"[Test] Episode: {episode}, Total Reward: {episode_reward:.2f}")
        if current_platform["level"] == 4:
            print('Reached the Top Platform!')
        else:
            print('Not reach the top yet')
        current_score = episode_reward
        if episode_reward > high_score:
            high_score = episode_reward

        log_test_episode(episode, episode_reward, q_agent.epsilon)

        episode += 1
        pygame.time.wait(1000)

def plot_test_curve_and_save():
    """
    Read the test log file, extract episodes, total rewards, and epsilon values,
    then plot and save both the total reward curve and the epsilon decay curve.
    Additionally, compute and display statistical metrics like mean reward, standard deviation,
    and moving average reward on the test reward curve.
    """
    episodes = []
    total_rewards = []
    epsilons = []

    try:
        with open(test_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Expected log format: "Test Episode: 10, Total Reward: -20.30, Epsilon: 0.050"
                m = re.search(r"Test Episode:\s*(\d+),\s*Total Reward:\s*([-+]?\d*\.?\d+),\s*Epsilon:\s*([-+]?\d*\.?\d+)", line)
                if m:
                    episodes.append(int(m.group(1)))
                    total_rewards.append(float(m.group(2)))
                    epsilons.append(float(m.group(3)))
    except Exception as e:
        print("Read test log file error:", e)
        return

    if not episodes:
        print("No data found in the test log file.")
        return

    MOVING_AVG_WINDOW = 50
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    moving_avg_rewards = pd.Series(total_rewards).rolling(window=MOVING_AVG_WINDOW, min_periods=1).mean()

    # --- Plot Test Total Reward Curve ---
    plt.figure(figsize=(10, 5))
    # Plot raw reward curve
    plt.plot(episodes, total_rewards, marker='o', color='green', label='Test Total Reward', alpha=0.5)
    # Plot horizontal line for mean reward
    plt.axhline(y=mean_reward, color='blue', linestyle='--', label=f'Mean Reward ({mean_reward:.2f})')
    # Plot moving average reward curve
    plt.plot(episodes, moving_avg_rewards, color='red', linestyle='-', linewidth=2, label='Moving Avg Reward')

    plt.xlabel('Test Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Total Reward Curve')
    plt.legend()
    plt.grid(True)

    stats_text = f"Mean Reward: {mean_reward:.2f}\nStd Dev: {std_reward:.2f}"
    plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))

    test_learning_curve_path = os.path.join(save_dir, "test_learning_curve.png")
    plt.savefig(test_learning_curve_path)
    print(f"Test total reward curve saved to: {test_learning_curve_path}")
    plt.close()

    # --- Plot Test Epsilon Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, marker='x', color='blue', label='Test Epsilon')
    plt.xlabel('Test Episode')
    plt.ylabel('Epsilon')
    plt.title('Test Epsilon Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    test_epsilon_curve_path = os.path.join(save_dir, "test_epsilon_curve.png")
    plt.savefig(test_epsilon_curve_path)
    print(f"Test epsilon curve saved to: {test_epsilon_curve_path}")
    plt.close()


if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("Testing stopped by user")
    finally:
        print("Start to plot the learning curve and save the Q-table")
        plot_test_curve_and_save()
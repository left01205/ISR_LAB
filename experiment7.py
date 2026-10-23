import pygame
import math
import numpy as np

# -------------------------------------------------------------
# 1. PYGAME & FIELD CONFIGURATION
# -------------------------------------------------------------
pygame.init()
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Exp 15: Autonomous 2v2 Robot Soccer Simulation")
clock = pygame.time.Clock()

# Colors
FIELD_GREEN = (34, 139, 34)
LINE_WHITE = (240, 240, 240)
BLUE_TEAM = (50, 120, 255)
RED_TEAM = (255, 60, 60)
BALL_COLOR = (255, 255, 255)
GOAL_COLOR = (255, 215, 0)
DARK_BG = (20, 20, 20)

font = pygame.font.SysFont("Segoe UI", 16, bold=True)
score_font = pygame.font.SysFont("Segoe UI", 28, bold=True)

# -------------------------------------------------------------
# 2. GAME STATE & AGENT DEFINITION
# -------------------------------------------------------------
class RobotAgent:
    def __init__(self, x, y, team, role):
        self.home_x = x
        self.home_y = y
        self.pos = [float(x), float(y)]
        self.team = team  # 'blue' or 'red'
        self.role = role  # 'striker' or 'defender'
        self.radius = 16
        self.speed = 3.2 if role == 'striker' else 2.5

    def reset(self):
        self.pos = [float(self.home_x), float(self.home_y)]

    def update(self, ball_pos, goal_target):
        # AI Strategy: Striker chases ball directly; Defender stays near home X
        if self.role == 'striker':
            target_x, target_y = ball_pos[0], ball_pos[1]
        else:
            # Defender stays between ball and own goal
            target_x = self.home_x + (ball_pos[0] - self.home_x) * 0.3
            target_y = ball_pos[1]

        dx = target_x - self.pos[0]
        dy = target_y - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist > 3:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed

# Initialize Pitch Entities
blue_robots = [
    RobotAgent(200, 250, 'blue', 'striker'),
    RobotAgent(100, 250, 'blue', 'defender')
]

red_robots = [
    RobotAgent(700, 250, 'red', 'striker'),
    RobotAgent(800, 250, 'red', 'defender')
]

ball_pos = [450.0, 250.0]
ball_vel = [0.0, 0.0]
ball_radius = 9

blue_score = 0
red_score = 0

def reset_positions():
    global ball_pos, ball_vel
    ball_pos = [450.0, 250.0]
    ball_vel = [0.0, 0.0]
    for r in blue_robots + red_robots:
        r.reset()

running = True
while running:
    screen.fill(FIELD_GREEN)

    # -------------------------------------------------------------
    # 3. EVENT HANDLING (DRAG BALL TO INTERACT)
    # -------------------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
            mx, my = event.pos
            ball_pos = [float(mx), float(my)]
            ball_vel = [0.0, 0.0]

    # -------------------------------------------------------------
    # 4. AI AGENT UPDATE & BALL KICK MECHANICS
    # -------------------------------------------------------------
    blue_goal_target = [870, 250]  # Attack Right
    red_goal_target = [30, 250]     # Attack Left

    for r in blue_robots:
        r.update(ball_pos, blue_goal_target)
    for r in red_robots:
        r.update(ball_pos, red_goal_target)

    # Check collisions between robots and ball
    all_robots = [(r, blue_goal_target) for r in blue_robots] + [(r, red_goal_target) for r in red_robots]
    
    for r, target in all_robots:
        dx = ball_pos[0] - r.pos[0]
        dy = ball_pos[1] - r.pos[1]
        dist = math.hypot(dx, dy)

        if dist < (r.radius + ball_radius):
            # Kick ball toward designated goal target
            kick_dx = target[0] - ball_pos[0]
            kick_dy = target[1] - ball_pos[1]
            kick_dist = math.hypot(kick_dx, kick_dy)
            
            kick_power = 9.0
            ball_vel[0] = (kick_dx / kick_dist) * kick_power
            ball_vel[1] = (kick_dy / kick_dist) * kick_power

    # Update Ball Position & Apply Ground Friction
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]
    ball_vel[0] *= 0.94
    ball_vel[1] *= 0.94

    # Pitch Boundary Bounce Mechanics
    if ball_pos[1] <= 45 or ball_pos[1] >= 455:
        ball_vel[1] *= -1.0

    # -------------------------------------------------------------
    # 5. GOAL DETECTION LOGIC
    # -------------------------------------------------------------
    if 170 <= ball_pos[1] <= 330:
        if ball_pos[0] >= 870:  # Blue Scores!
            blue_score += 1
            reset_positions()
        elif ball_pos[0] <= 30:  # Red Scores!
            red_score += 1
            reset_positions()
    else:
        # Bounce off back wall outside goal post
        if ball_pos[0] <= 30 or ball_pos[0] >= 870:
            ball_vel[0] *= -1.0

    # Clamp ball position inside field limits
    ball_pos[0] = np.clip(ball_pos[0], 25, 875)
    ball_pos[1] = np.clip(ball_pos[1], 25, 475)

    # -------------------------------------------------------------
    # 6. FIELD RENDERING & DASHBOARD OVERLAY
    # -------------------------------------------------------------
    # Pitch Boundaries & Lines
    pygame.draw.rect(screen, LINE_WHITE, (30, 30, 840, 440), 3)
    pygame.draw.line(screen, LINE_WHITE, (450, 30), (450, 470), 2)
    pygame.draw.circle(screen, LINE_WHITE, (450, 250), 70, 2)

    # Goals (Left & Right)
    pygame.draw.rect(screen, GOAL_COLOR, (10, 170, 20, 160), 3)
    pygame.draw.rect(screen, GOAL_COLOR, (870, 170, 20, 160), 3)

    # Draw Robots
    for r in blue_robots:
        pygame.draw.circle(screen, BLUE_TEAM, (int(r.pos[0]), int(r.pos[1])), r.radius)
        pygame.draw.circle(screen, LINE_WHITE, (int(r.pos[0]), int(r.pos[1])), r.radius, 2)

    for r in red_robots:
        pygame.draw.circle(screen, RED_TEAM, (int(r.pos[0]), int(r.pos[1])), r.radius)
        pygame.draw.circle(screen, LINE_WHITE, (int(r.pos[0]), int(r.pos[1])), r.radius, 2)

    # Draw Ball
    pygame.draw.circle(screen, BALL_COLOR, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)
    pygame.draw.circle(screen, DARK_BG, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, 1)

    # Dashboard Scoreboard
    score_txt = score_font.render(f"BLUE {blue_score}  -  {red_score} RED", True, LINE_WHITE)
    screen.blit(score_txt, (WIDTH // 2 - score_txt.get_width() // 2, 5))

    info_txt = font.render("[Click & Drag] Reposition Ball Live", True, LINE_WHITE)
    screen.blit(info_txt, (20, 475))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

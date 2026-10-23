import pygame, math, numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 450))
pygame.display.set_caption("Exp 7: 2v2 Robot Soccer Simulation")
clock, font = pygame.time.Clock(), pygame.font.SysFont("Arial", 24, bold=True)

# Robots: [x, y, home_x, color, role, goal_x]
BLUE, RED = (50, 120, 255), (255, 60, 60)
robots = [
    [200.0, 225.0, 200, BLUE, 'striker', 780],
    [100.0, 225.0, 100, BLUE, 'defender', 780],
    [600.0, 225.0, 600, RED,  'striker', 20],
    [700.0, 225.0, 700, RED,  'defender', 20]
]
ball, vel, score = [400.0, 225.0], [0.0, 0.0], [0, 0]

running = True
while running:
    screen.fill((34, 139, 34))
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif pygame.mouse.get_pressed()[0]: ball, vel = list(e.pos), [0.0, 0.0]

    # Update AI Agents & Kick Mechanics
    for r in robots:
        tx = ball[0] if r[4] == 'striker' else r[2] + (ball[0] - r[2]) * 0.3
        dx, dy = tx - r[0], ball[1] - r[1]
        d = math.hypot(dx, dy)
        if d > 3:
            spd = 3.2 if r[4] == 'striker' else 2.4
            r[0] += (dx / d) * spd; r[1] += (dy / d) * spd

        if math.hypot(ball[0] - r[0], ball[1] - r[1]) < 24:
            kx, ky = r[5] - ball[0], 225 - ball[1]
            kd = math.hypot(kx, ky)
            vel = [(kx / kd) * 9.0, (ky / kd) * 9.0]

        pygame.draw.circle(screen, r[3], (int(r[0]), int(r[1])), 15)

    # Ball Physics & Goal Scoring
    ball[0] += vel[0]; ball[1] += vel[1]
    vel[0] *= 0.94; vel[1] *= 0.94
    if ball[1] <= 30 or ball[1] >= 420: vel[1] *= -1

    if 150 <= ball[1] <= 300 and (ball[0] <= 25 or ball[0] >= 775):
        score[0 if ball[0] >= 775 else 1] += 1
        ball, vel = [400.0, 225.0], [0.0, 0.0]
    elif ball[0] <= 25 or ball[0] >= 775:
        vel[0] *= -1

    pygame.draw.rect(screen, (255, 255, 255), (20, 20, 760, 410), 2)
    pygame.draw.circle(screen, (255, 255, 255), (int(ball[0]), int(ball[1])), 9)
    screen.blit(font.render(f"BLUE {score[0]} - {score[1]} RED", True, (255, 255, 255)), (310, 25))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PID Motor Position Control Simulation")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (30, 30, 30)
WHITE = (240, 240, 240)
RED = (255, 75, 75)
BLUE = (75, 150, 255)
GREEN = (50, 205, 50)
GRAY = (100, 100, 100)

# Physical Motor Parameters
angle = 0.0          # Current motor angle (degrees)
angular_vel = 0.0    # Angular velocity
target_angle = 90.0  # Target angle setpoint (degrees)

# PID Controller Parameters
Kp = 0.08
Ki = 0.002
Kd = 0.05

integral = 0.0
prev_error = 0.0

# History tracking for live plotting
history_len = 300
actual_history = []
target_history = []

font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 20, bold=True)

running = True
dt = 0.1

while running:
    screen.fill(BG_COLOR)
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
            # Drag or click on left half to manually set Target Angle
            mx, my = event.pos
            if mx < 450:
                dx = mx - 225
                dy = my - 250
                target_angle = math.degrees(math.atan2(dy, dx)) % 360

    # 1. PID Calculation
    error = target_angle - angle
    # Handle angle wrap-around shortest path (-180 to 180)
    error = (error + 180) % 360 - 180
    
    integral += error * dt
    derivative = (error - prev_error) / dt
    prev_error = error
    
    # Control Torque / Voltage Output
    torque = (Kp * error) + (Ki * integral) + (Kd * derivative)
    torque = np.clip(torque, -15, 15)  # Motor physical limits
    
    # 2. Motor Dynamics Update
    damping = 0.05
    angular_acc = torque - (damping * angular_vel)
    angular_vel += angular_acc * dt
    angle = (angle + angular_vel * dt) % 360

    # Store tracking history
    actual_history.append(angle)
    target_history.append(target_angle)
    if len(actual_history) > history_len:
        actual_history.pop(0)
        target_history.pop(0)

    # -------------------------------------------------------------
    # DRAWING PANEL 1: 2D MOTOR ARM & TARGET DIAL (LEFT SIDE)
    # -------------------------------------------------------------
    center_x, center_y = 225, 250
    arm_length = 120

    # Section Title
    txt = title_font.render("Interactive Motor Shaft", True, WHITE)
    screen.blit(txt, (center_x - txt.get_width()//2, 20))

    # Outer dial ring
    pygame.draw.circle(screen, GRAY, (center_x, center_y), arm_length + 10, 2)

    # Target Angle Line (Red Dashed Line)
    target_rad = math.radians(target_angle)
    tx = center_x + arm_length * math.cos(target_rad)
    ty = center_y + arm_length * math.sin(target_rad)
    pygame.draw.line(screen, RED, (center_x, center_y), (tx, ty), 3)
    pygame.draw.circle(screen, RED, (int(tx), int(ty)), 6)

    # Actual Motor Arm (Blue Solid Line)
    actual_rad = math.radians(angle)
    ax = center_x + arm_length * math.cos(actual_rad)
    ay = center_y + arm_length * math.sin(actual_rad)
    pygame.draw.line(screen, BLUE, (center_x, center_y), (ax, ay), 6)
    pygame.draw.circle(screen, GREEN, (center_x, center_y), 15)  # Motor Hub

    # Readouts
    info1 = font.render(f"Target Angle: {target_angle:.1f}° (Click/Drag to change)", True, RED)
    info2 = font.render(f"Actual Angle: {angle:.1f}°", True, BLUE)
    info3 = font.render(f"Applied Torque: {torque:.2f} Nm", True, WHITE)
    screen.blit(info1, (30, 410))
    screen.blit(info2, (30, 435))
    screen.blit(info3, (30, 460))

    # -------------------------------------------------------------
    # DRAWING PANEL 2: REAL-TIME SCOPE (RIGHT SIDE)
    # -------------------------------------------------------------
    plot_x, plot_y = 480, 70
    plot_w, plot_h = 380, 360

    txt2 = title_font.render("Live Angle Tracking Oscilloscope", True, WHITE)
    screen.blit(txt2, (plot_x + plot_w//2 - txt2.get_width()//2, 20))

    pygame.draw.rect(screen, (15, 15, 15), (plot_x, plot_y, plot_w, plot_h))
    pygame.draw.rect(screen, GRAY, (plot_x, plot_y, plot_w, plot_h), 1)

    # Plot lines
    if len(actual_history) > 1:
        for i in range(1, len(actual_history)):
            x1 = plot_x + (i - 1) * (plot_w / history_len)
            x2 = plot_x + i * (plot_w / history_len)

            # Map 0..360 degrees to plot height
            y_target1 = plot_y + plot_h - (target_history[i-1] / 360.0 * plot_h)
            y_target2 = plot_y + plot_h - (target_history[i] / 360.0 * plot_h)
            y_act1 = plot_y + plot_h - (actual_history[i-1] / 360.0 * plot_h)
            y_act2 = plot_y + plot_h - (actual_history[i] / 360.0 * plot_h)

            pygame.draw.line(screen, RED, (x1, y_target1), (x2, y_target2), 2)
            pygame.draw.line(screen, BLUE, (x1, y_act1), (x2, y_act2), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import math

# -------------------------------------------------------------
# 1. PYTORCH DATASET & MODEL DEFINITION
# -------------------------------------------------------------
np.random.seed(42)
L1, L2 = 110.0, 90.0  # Link lengths in pixels

def analytical_ik(x, y):
    """Exact geometric Inverse Kinematics (Ground Truth)."""
    dist = math.hypot(x, y)
    dist = max(10.0, min(dist, L1 + L2 - 1.0))
    cos_theta2 = (dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = math.acos(cos_theta2)
    k1 = L1 + L2 * math.cos(theta2)
    k2 = L2 * math.sin(theta2)
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)
    return theta1, theta2

# Generate synthetic dataset: Target [X, Y] -> Joint Angles [Theta1, Theta2]
num_samples = 1000
X_raw = np.random.uniform(-180, 180, (num_samples, 2))
Y_raw = np.zeros((num_samples, 2))
for i in range(num_samples):
    Y_raw[i, 0], Y_raw[i, 1] = analytical_ik(X_raw[i, 0], X_raw[i, 1])

SCALE_XY = L1 + L2
X_tensor = torch.FloatTensor(X_raw / SCALE_XY)
Y_tensor = torch.FloatTensor(Y_raw)

# Neural Network Architecture
class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

model = InverseKinematicsNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.008)

# -------------------------------------------------------------
# 2. PYGAME DASHBOARD SETUP
# -------------------------------------------------------------
pygame.init()
WIDTH, HEIGHT = 950, 520
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Exp 9: PyTorch Neural Network Kinematics Controller")
clock = pygame.time.Clock()

BG_COLOR = (20, 22, 28)
PANEL_BG = (30, 34, 45)
WHITE = (240, 240, 240)
RED = (255, 80, 80)
BLUE = (60, 160, 255)
GREEN = (50, 215, 120)
GRAY = (90, 95, 110)

font = pygame.font.SysFont("Segoe UI", 15)
font_bold = pygame.font.SysFont("Segoe UI", 16, bold=True)
title_font = pygame.font.SysFont("Segoe UI", 19, bold=True)

arm_origin = (220, 270)
target_screen = [310, 190]
epoch_count = 0
loss_history = []
auto_train = True

def train_step(steps=2):
    global epoch_count
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, Y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        epoch_count += 1

running = True
while running:
    screen.fill(BG_COLOR)
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                auto_train = not auto_train
            elif event.key == pygame.K_t:
                train_step(50)
        elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
            mx, my = event.pos
            if mx < 430:  # Set target position inside left panel
                target_screen = [mx, my]

    if auto_train and epoch_count < 400:
        train_step(2)

    # -------------------------------------------------------------
    # 3. KINEMATICS & INFERENCE
    # -------------------------------------------------------------
    rel_x = target_screen[0] - arm_origin[0]
    rel_y = target_screen[1] - arm_origin[1]
    
    # Ground Truth Angles vs NN Predicted Angles
    gt_th1, gt_th2 = analytical_ik(rel_x, rel_y)
    
    model.eval()
    with torch.no_grad():
        inp = torch.FloatTensor([[rel_x / SCALE_XY, rel_y / SCALE_XY]])
        pred_angles = model(inp).numpy()[0]
        nn_th1, nn_th2 = pred_angles[0], pred_angles[1]

    # Ghost Arm Positions (Math Exact)
    gt_j1 = (arm_origin[0] + L1 * math.cos(gt_th1), arm_origin[1] + L1 * math.sin(gt_th1))
    gt_j2 = (gt_j1[0] + L2 * math.cos(gt_th1 + gt_th2), gt_j1[1] + L2 * math.sin(gt_th1 + gt_th2))

    # Neural Network Predicted Arm Positions
    nn_j1 = (arm_origin[0] + L1 * math.cos(nn_th1), arm_origin[1] + L1 * math.sin(nn_th1))
    nn_j2 = (nn_j1[0] + L2 * math.cos(nn_th1 + nn_th2), nn_j1[1] + L2 * math.sin(nn_th1 + nn_th2))

    end_error = math.hypot(nn_j2[0] - target_screen[0], nn_j2[1] - target_screen[1])

    # -------------------------------------------------------------
    # PANEL 1: PHYSICAL ARM & TARGET (LEFT SIDE)
    # -------------------------------------------------------------
    pygame.draw.rect(screen, PANEL_BG, (15, 15, 410, 480), border_radius=10)
    screen.blit(title_font.render("2-DOF Robotic Arm Control", True, WHITE), (30, 25))

    # Workspace Circle Boundary
    pygame.draw.circle(screen, (40, 45, 60), arm_origin, int(L1 + L2), 1)

    # Draw Ground Truth Ghost Arm (Dashed Gray)
    pygame.draw.line(screen, GRAY, arm_origin, gt_j1, 2)
    pygame.draw.line(screen, GRAY, gt_j1, gt_j2, 2)

    # Draw NN Predicted Arm (Solid Blue & Green)
    pygame.draw.line(screen, BLUE, arm_origin, nn_j1, 6)
    pygame.draw.line(screen, GREEN, nn_j1, nn_j2, 5)
    
    # Joints & Base
    pygame.draw.circle(screen, WHITE, arm_origin, 10)
    pygame.draw.circle(screen, WHITE, (int(nn_j1[0]), int(nn_j1[1])), 7)
    pygame.draw.circle(screen, GREEN, (int(nn_j2[0]), int(nn_j2[1])), 6)

    # Target Indicator (Red Ring)
    pygame.draw.circle(screen, RED, target_screen, 9, 2)
    pygame.draw.circle(screen, RED, target_screen, 3)

    # Readouts
    screen.blit(font.render(f"Target Pos (X,Y): [{rel_x:.0f}, {rel_y:.0f}]", True, RED), (30, 410))
    screen.blit(font.render(f"Position Error: {end_error:.1f} px", True, GREEN if end_error < 15 else RED), (30, 435))
    screen.blit(font.render("Gray Line = Exact Math | Solid Line = NN Output", True, GRAY), (30, 460))

    # -------------------------------------------------------------
    # PANEL 2: PYTORCH TRAINING DASHBOARD (RIGHT SIDE)
    # -------------------------------------------------------------
    plot_x, plot_y = 440, 15
    plot_w, plot_h = 495, 480
    pygame.draw.rect(screen, PANEL_BG, (plot_x, plot_y, plot_w, plot_h), border_radius=10)

    screen.blit(title_font.render("PyTorch Model Convergence", True, WHITE), (plot_x + 20, 25))

    # Training Telemetry
    screen.blit(font.render(f"Epochs Trained: {epoch_count}", True, WHITE), (plot_x + 20, 60))
    screen.blit(font.render(f"Loss (MSE): {loss_history[-1]:.6f}" if loss_history else "Loss: --", True, GREEN), (plot_x + 220, 60))

    # Loss Curve Scope
    scope_x, scope_y = plot_x + 20, plot_y + 90
    scope_w, scope_h = 455, 230
    pygame.draw.rect(screen, (15, 17, 23), (scope_x, scope_y, scope_w, scope_h), border_radius=6)
    pygame.draw.rect(screen, GRAY, (scope_x, scope_y, scope_w, scope_h), 1, border_radius=6)

    if len(loss_history) > 1:
        max_l = max(loss_history[:10]) if max(loss_history[:10]) > 0 else 1.0
        pts = []
        for idx, l_val in enumerate(loss_history):
            px = scope_x + (idx / len(loss_history)) * scope_w
            py = scope_y + scope_h - (min(l_val, max_l) / max_l) * (scope_h - 20) - 10
            pts.append((px, py))
        pygame.draw.lines(screen, GREEN, False, pts, 2)

    # Angle Comparison Table
    table_y = scope_y + scope_h + 20
    screen.blit(font_bold.render("Joint Angles", True, WHITE), (plot_x + 20, table_y))
    screen.blit(font_bold.render("Exact Math", True, GRAY), (plot_x + 180, table_y))
    screen.blit(font_bold.render("NN Predicted", True, BLUE), (plot_x + 330, table_y))

    screen.blit(font.render("Joint 1 (θ1):", True, WHITE), (plot_x + 20, table_y + 25))
    screen.blit(font.render(f"{math.degrees(gt_th1):.1f}°", True, GRAY), (plot_x + 180, table_y + 25))
    screen.blit(font.render(f"{math.degrees(nn_th1):.1f}°", True, BLUE), (plot_x + 330, table_y + 25))

    screen.blit(font.render("Joint 2 (θ2):", True, WHITE), (plot_x + 20, table_y + 50))
    screen.blit(font.render(f"{math.degrees(gt_th2):.1f}°", True, GRAY), (plot_x + 180, table_y + 50))
    screen.blit(font.render(f"{math.degrees(nn_th2):.1f}°", True, BLUE), (plot_x + 330, table_y + 50))

    screen.blit(font.render("[Click/Drag] Move Target  |  [SPACE] Pause Training  |  [T] Train +50", True, GRAY), (plot_x + 20, table_y + 90))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

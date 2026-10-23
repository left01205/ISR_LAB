import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pygame, math

L1, L2 = 110.0, 90.0

def analytical_ik(x, y):
    d = np.clip(math.hypot(x, y), 25.0, L1 + L2 - 1.0)
    c2 = np.clip((d**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    t2 = math.acos(c2)
    t1 = math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * c2)
    return t1, t2

# 1. Generate Dataset & Train PyTorch Neural Network
X_raw = np.random.uniform(-180, 180, (1000, 2))
Y_raw = np.array([analytical_ik(x, y) for x, y in X_raw])
X = torch.FloatTensor(X_raw / 200.0)
Y = torch.FloatTensor(Y_raw)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(300):
    optimizer.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    optimizer.step()
print(f"Training Complete | Final MSE Loss: {loss.item():.4f}")

# 2. Interactive Pygame Robotic Arm Control
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Exp 6: NN Robotic Arm Controller")
origin, target = (250, 250), [350, 180]

running = True
while running:
    screen.fill((25, 25, 30))
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif pygame.mouse.get_pressed()[0]: target = list(e.pos)

    rx, ry = target[0] - origin[0], target[1] - origin[1]
    with torch.no_grad():
        t1, t2 = model(torch.FloatTensor([[rx / 200.0, ry / 200.0]]))[0].numpy()

    j1 = (origin[0] + L1 * math.cos(t1), origin[1] + L1 * math.sin(t1))
    j2 = (j1[0] + L2 * math.cos(t1 + t2), j1[1] + L2 * math.sin(t1 + t2))

    pygame.draw.line(screen, (60, 160, 255), origin, j1, 6)
    pygame.draw.line(screen, (50, 215, 120), j1, j2, 5)
    pygame.draw.circle(screen, (255, 80, 80), target, 8)
    pygame.display.flip()

pygame.quit()
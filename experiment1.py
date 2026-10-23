import matplotlib
matplotlib.use('TkAgg')  # Force interactive GUI window
import matplotlib.pyplot as plt
import numpy as np
import heapq

# Enable interactive plotting mode
plt.ion()

# 1. Grid setup: 0 = free space, 1 = obstacle
grid = np.zeros((10, 10))
grid[2:8, 5] = 1  # Obstacle wall in the middle

start = (0, 0)
goal = (9, 9)

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def visual_astar(grid, start, goal, delay=0.1):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    # Setup figure and initial plot
    fig, ax = plt.subplots(figsize=(7, 7))

    while oheap:
        current = heapq.heappop(oheap)[1]

        # Redraw the grid status at current decision step
        ax.clear()
        
        # Base grid: white = free space, black = obstacle
        ax.imshow(grid, cmap='binary')

        # Draw Closed Set (Evaluated / Visited nodes) -> Light Blue
        if close_set:
            cx, cy = zip(*close_set)
            ax.scatter(cy, cx, color='skyblue', s=120, label='Visited (Closed Set)')

        # Draw Open Set (Frontier nodes waiting in queue) -> Yellow
        open_nodes = [item[1] for item in oheap]
        if open_nodes:
            ox, oy = zip(*open_nodes)
            ax.scatter(oy, ox, color='orange', s=120, label='Frontier (Open Set)')

        # Draw Current Active Node -> Red
        ax.scatter(current[1], current[0], color='red', s=180, label='Current Node')

        # Draw Start and Goal
        ax.scatter(start[1], start[0], color='green', marker='s', s=200, label='Start')
        ax.scatter(goal[1], goal[0], color='purple', marker='*', s=250, label='Goal')

        # Format plot
        ax.set_title(f"A* Decision Step: Evaluating Cell ({current[0]}, {current[1]})", fontsize=12)
        ax.set_xticks(np.arange(-0.5, 10, 1))
        ax.set_yticks(np.arange(-0.5, 10, 1))
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.draw()
        plt.pause(delay)  # Pause execution briefly to show frame

        # Reached target
        if current == goal:
            # Reconstruct shortest path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]

            # Draw final optimal path line
            px, py = zip(*path)
            ax.plot(py, px, color='lime', linewidth=4, label='Final Path')
            ax.set_title("Goal Reached! Optimal Path Constructed.", fontsize=12, color='green')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep final plot open
            return path

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [n[1] for n in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    plt.ioff()
    plt.show()
    return None

# Run visualized algorithm (adjust delay parameter for faster/slower animation)
visual_astar(grid, start, goal, delay=0.15)


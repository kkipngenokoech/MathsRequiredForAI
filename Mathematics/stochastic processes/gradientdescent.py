import numpy as np

def elevation(x, y):
    return 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)

def gradient(x, y):
    dx = 2*x + 20*np.pi*np.sin(2*np.pi*x)
    dy = 2*y + 20*np.pi*np.sin(2*np.pi*y)
    return np.array([dx, dy])

def gradient_ascent(start_x, start_y, learning_rate, num_steps):
    x, y = start_x, start_y
    for _ in range(num_steps):
        grad = gradient(x, y)
        x += learning_rate * grad[0]
        y += learning_rate * grad[1]
    return x, y

# Parameters
start_x, start_y = -1.8, -0.2
learning_rate = 0.001
num_steps = 100

# Run simulation
final_x, final_y = gradient_ascent(start_x, start_y, learning_rate, num_steps)

# Print result
print(f"Final position: ({final_x:.1f}, {final_y:.1f})")
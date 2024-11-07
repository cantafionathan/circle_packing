import numpy as np
import matplotlib.pyplot as plt

def greedy_pack_squares(N):
    # Calculate the grid size (number of rows and columns)
    grid_size = int(np.ceil(np.sqrt(N)))  # Square root of N, rounded up
    spacing = 1.05  # Small spacing to avoid overlap between squares

    squares = []
    for i in range(N):
        row = i // grid_size
        col = i % grid_size
        x_pos = col * spacing - (grid_size - 1) * spacing / 2
        y_pos = row * spacing - (grid_size - 1) * spacing / 2
        rotation = 0  # No rotation for simplicity
        squares.append([x_pos, y_pos, rotation])

    # Estimate enclosing square side length S as the span needed to fit all rows and columns
    S = grid_size * spacing

    return np.array(squares), S

# return the signed distance from point x,y to square with center and angle stored in [u, v, theta]
def signed_distance(square, x, y, r = 0.5 * 2**0.5):
    xy = np.stack([x.ravel(), y.ravel()], axis=-1) 
    center = np.array(square[:2])
    theta = square[2] + np.pi/4

    # rotate to be parallel with horizontal
    R = [[np.cos(-theta), -np.sin(-theta)], 
        [np.sin(-theta), np.cos(-theta)]]
    xy = (xy - center) @ R
    
    # compute distance to ball using p-norm
    distances = np.linalg.norm(xy, axis = -1, ord = 1) - r
    return distances.reshape(x.shape)

# plot a single square using zero contour of signed distance
def plot_square(square, ax, x_range = 10, y_range = 10, col = 'black', r = 0.5 * 2**0.5):
    feature_x = np.arange(-x_range, x_range, 0.01)
    feature_y = np.arange(-y_range, y_range, 0.01)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = [0] * len(X)
    Z = signed_distance(square, X, Y, r = r)
    ax.contour(X, Y, Z, levels=[0], colors = col)

def plot_squares(squares, S, ax):
    for s in squares:
        plot_square(s, ax)
    plot_square(np.array([0, 0, 0]), ax, col = 'blue', r = S/2 * 2**0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-S/2-1, S/2+1)
    ax.set_ylim(-S/2-1, S/2+1)

N = 17  # Number of squares to pack
squares, S = greedy_pack_squares(N)

# Plot the packed squares
fig, ax = plt.subplots()
plot_squares(squares, S, ax)
plt.show()
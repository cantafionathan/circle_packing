import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# check overlap:
# Separating axis theorem: two squares overlap if they overlap on every projection onto their normal planes

# rotate point (x,y) around point (u, v)
def rotate(x, y, u, v, theta):
    x = x - u
    y = y - v
    # rotate to be parallel with horizontal
    R = [[np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]]
    xy = np.dot(R, [x, y])
    xy[0] = xy[0] + u
    xy[1] = xy[1] + v
    return xy

def get_vertices(rect):
    x, y, theta = rect
    s = 1 # side length 
    # (unit 1-balls have side-length sqrt(2))
    # (unit squares have 1-ball radius 1/sqrt(2))
    
    vertices = [
        np.array([x - s/2, y - s/2]),  # Bottom-left
        np.array([x + s/2, y - s/2]),  # Bottom-right
        np.array([x + s/2, y + s/2]),  # Top-right
        np.array([x - s/2, y + s/2])   # Top-left
    ]

    vertices = [rotate(v[0], v[1], x, y, theta) for v in vertices]
    
    return np.array(vertices)

def get_axes(rect):
        return [(rect[i][1] - rect[i-1][1], rect[i-1][0] - rect[i][0]) for i in range(4)]

def project(rect, axis):
    # Project each vertex of the rectangle onto the axis
    return [np.dot(vertex, axis) for vertex in rect]

def overlap(proj1, proj2):
    # Check if projections overlap and by how much
    return min(max(proj1) - min(proj2), max(proj2) - min(proj1))
    
# check rect1 = (x1, y1, theta1) and rect2 = (x2, y2, theta2) do not overlap:
# use seperating axis theorem to check all the normal planes, project onto those planes and show seperation in at least one
# if cannot find seperation then they intersect
def check_overlap(rect1, rect2):
    rect1 = get_vertices(rect1)
    rect2 = get_vertices(rect2)
    
    axes1 = get_axes(rect1)
    axes2 = get_axes(rect2)

    new_overlap = float('inf')
    for axis in axes1 + axes2:
        proj1 = project(rect1, axis)
        proj2 = project(rect2, axis)

        current_overlap = overlap(proj1, proj2)

        if current_overlap >= 0:
            return current_overlap # separting axis found, return the postive separation distance
        
        new_overlap = min(current_overlap, new_overlap)
        
    return new_overlap  # No separating axis found, return the extent of the overlap

# check contained
def check_contained(rect, S):
    vertices = get_vertices(rect)
    
    # Calculate the minimum margin by which each vertex is within bounds
    margin_x = S/2 - np.abs(vertices[:, 0])
    margin_y = S/2 - np.abs(vertices[:, 1])
    
    # Return the smallest containment margin (positive if inside)
    return min(np.min(margin_x), np.min(margin_y))

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

def plot_squares(x, N, ax):
    ax.clear()  # Clear the current plot
    squares = x[:3 * N].reshape((N, 3))
    S = x[-1]
    for i in range(N):
        plot_square(squares[i], ax)
    plot_square(np.array([0, 0, 0]), ax, col = 'blue', r = S/2 * 2**0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-S/2-1, S/2+1)
    ax.set_ylim(-S/2-1, S/2+1)
    ax.set_title(f"Enclosing Square Side Length: {S:.2f}")

frames = []

def pack_squares(N):
    # greedy algorithm to initialize the squares
    # Calculate the grid size (number of rows and columns)
    grid_size = int(np.ceil(np.sqrt(N)))  # Square root of N, rounded up
    spacing = 1.00  # Small spacing to avoid overlap between squares

    x0 = []
    for i in range(N):
        row = i // grid_size
        col = i % grid_size
        u = col * spacing - (grid_size - 1) * spacing / 2 # makes it better for n > 1 COBYLA, but worse for n = 1?
        v = row * spacing - (grid_size - 1) * spacing / 2 # COBYLA doesn't even converge for n = 1???
        theta = 0  # No rotation for simplicity
        x0.extend([u, v, theta])

    # Estimate enclosing square side length S as the span needed to fit all rows and columns
    S0 = grid_size * spacing
    x0.append(S0)

    def objective(x):
        return x[-1]
    
    constraints = []

    # non-overlapping constraints
    for i in range(N):
        for j in range(1 + i, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: check_overlap(x[3*i:3*i + 3], x[3*j:3*j + 3])
            })

    # contained in enclosing square
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: check_contained(x[3*i:3*i + 3], x[-1])
        })

    # # fixed angle constraint for testing
    # for i in range(N):
    #     constraints.append({
    #         'type': 'ineq',
    #         'fun': lambda x, i=i: x[3*i + 2]
    #     })

    # for i in range(N):
    #     constraints.append({
    #         'type': 'ineq',
    #         'fun': lambda x, i=i: -x[3*i + 2]
    #     })

    # Set up the plot for displaying squares
    fig, ax = plt.subplots() # for watching it go even if it fails

    # Callback function for updating plot during optimization
    def callback(x):
        # frames.append(x) # for saving animation
        plot_squares(x, N, ax) # for watching it go even if it fails
        plt.pause(0.01) # for watching it go even if it fails

    options = {'maxiter': 100, 'ftol': 1e-6}

    # COBYLA or SLSQP
    result = minimize(objective, x0, constraints=constraints, method='SLSQP', options=options, callback=callback)

    if result.success:
        x_opt = result.x
        squares = x_opt[:3 * N].reshape((N, 3))
        S = x_opt[-1]
        return squares, S
    else:
        raise ValueError("Optimization failed!")
    
N = 1
pack_squares(N)

fig, ax = plt.subplots()

def animate(i):
    plot_squares(frames[i], N, ax)

# Create animation and save as GIF
anim = FuncAnimation(fig, animate, frames=len(frames), interval=200)
plt.show()
# anim.save("circle_packing.gif", writer=PillowWriter(fps=5))
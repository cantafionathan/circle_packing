import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation, PillowWriter

# Plotting purposes

frames = []

def signed_distance(c, x, y, r=1):
    xy = np.stack([x, y], axis = -1)
    # compute distance to ball using p-norm
    return np.linalg.norm(c - xy, axis = -1, ord = 2) - r

# plot a single ball
def plot_ball(c, ax, x_range = 10, y_range = 10, r = 1):
    feature_x = np.arange(-x_range, x_range, 0.01)
    feature_y = np.arange(-y_range, y_range, 0.01)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = [0] * len(X)
    Z = signed_distance(c, X, Y, r)
    ax.contour(X, Y, Z, levels=[0], colors = 'black')

# plot list of balls
def plot_balls(circles, R, ax):
    ax.set_aspect('equal')
    ax.clear()

    plot_ball(np.array([0,0]), ax, r=R, x_range = R+1, y_range = R+1)

    for i in range(len(circles)):
        plot_ball(circles[i], ax, x_range = R+1, y_range = R+1)

    ax.set_title(f'Radius: {R:.2f}, N: {len(circles):.2f}')

    plt.draw()
    plt.pause(0.01)

# pack N unit balls in the smallest possible containing ball
def pack_circles(N, animate = False):
    x0 = np.zeros(2*N + 1)
    R0 = N/2
    x0[-1] = R0

    def objective(x):
        return x[-1]
    
    constraints = []

    # non-overlapping constraints, i.e. d(ci, cj) >= 2
    for i in range(N):
        for j in range(1 + i, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i + 2] - x[2*j:2*j + 2]) - 2
            })

    # contained in enclosing circle, i.e. d(ci, 0) <= R - 1
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[-1] - 1 - np.linalg.norm(x[2*i:2*i + 2])
        })

    def callback(xk):
        circles = xk[:2 * N].reshape((N, 2))
        R = xk[-1]
        # plot_balls(circles, R, ax)
        frames.append((circles, R))

    if animate:
        fig, ax = plt.subplots()
        # plt.ion()  # Turn on interactive mode for live updating
        result = minimize(objective, x0, constraints=constraints, method='SLSQP', callback=callback)
        # plt.ioff()  # Turn off interactive mode
        # plt.show()  # Display the final plot
    else:
        result = minimize(objective, x0, constraints=constraints, method='SLSQP')

    if result.success:
        x_opt = result.x
        circles = x_opt[:2 * N].reshape((N, 2))
        R = x_opt[-1]
        if animate == False:
            fig, ax = plt.subplots()
            plot_balls(circles, R, ax)
            plt.show()
        return circles, R
    else:
        raise ValueError("Optimization failed!")

pack_circles(10, animate=True)

fig, ax = plt.subplots()

def animate(i):
    circles, R = frames[i]
    plot_balls(circles, R, ax)

# Create animation and save as GIF
anim = FuncAnimation(fig, animate, frames=len(frames), interval=200)
#plt.show()
anim.save("circle_packing.gif", writer=PillowWriter(fps=5))

import matplotlib.pyplot as plt
import numpy as np

class ball:
    def __init__(self, u: float, v: float, radius: float, 
                angle: float = 0, p: float = 2):
        self.center = (u, v)
        self.radius = radius
        self.angle = angle
        self.norm = p

# computes the signed distance of the point (x,y) to the p-ball
# centered at (u, v) with radius r, rotated by angle above the horizontal
# signed distance is positive outside, negative inside, and zero on
# p default to 2, (circles)
def signed_distance(b: ball, x, y):
    u = b.center[0]
    v = b.center[1]
    r = b.radius
    theta = b.angle
    p = b.norm

    # rotate to be parallel with horizontal
    R = [[np.cos(-theta), -np.sin(-theta)], 
        [np.sin(-theta), np.cos(-theta)]]
    xy = np.dot(R, [x, y])
    x = xy[0]
    y = xy[1]

    # compute distance to ball using p-norm
    return((np.abs(x - u)**p + np.abs(y - v)**p)**(1/p) - r)

# draw each ball in the list
def plot_balls(balls, x_range = 5, y_range = 5):
    feature_x = np.arange(-x_range, x_range, 0.01)
    feature_y = np.arange(-y_range, y_range, 0.01)
    fig, ax = plt.subplots()
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = [0] * len(X)

    if balls is ball:
        for k in range(len(X)):
                Z[k] = signed_distance(balls, X[k], Y[k])
        ax.contour(X, Y, Z, levels=[0], colors = 'black')
    else:
        for b in balls:
            for k in range(len(X)):
                Z[k] = signed_distance(b, X[k], Y[k])
            ax.contour(X, Y, Z, levels=[0], colors = 'black')

    plt.show()

# example
c1 = ball(0, 0, 1, p = 0.5)
c2 = ball(1, 0, 2)
c3 = ball(0, 0, 2, p =1)
plot_balls((c1, c2, c3))
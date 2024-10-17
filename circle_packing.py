import matplotlib.pyplot as plt
import numpy as np

class circle:
    def __init__(self, x: float, y: float, radius: float):
        self.center = (x, y)
        self.radius = radius

# computes the signed distance of the point (x,y) to the circle
# centered at (u, v) with radius r
# signed distance is positive outside, negative inside, and zero on
def signed_distance(c: circle, x, y):
    u = c.center[0]
    v = c.center[1]
    r = c.radius
    return(((x - u)**2 + (y - v)**2)**0.5 - r)

# draw each circle in the list
def plot_circles(circles: list, x_range = 5, y_range = 5):
    feature_x = np.arange(-1*x_range, x_range, 0.1)
    feature_y = np.arange(-1*y_range, y_range, 0.1)
    fig, ax = plt.subplots()
    [X, Y] = np.meshgrid(feature_x, feature_y)

    for c in circles:
        Z = signed_distance(c, X, Y)
        ax.contour(X, Y, Z, levels=[0])

    plt.show()

# example
c1 = circle(0, 0, 0.5)
c2 = circle(1, 2, 1)
plot_circles((c1, c2))
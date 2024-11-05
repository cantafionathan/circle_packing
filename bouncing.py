import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from matplotlib.animation import FuncAnimation, PillowWriter
from copy import deepcopy

# the algorithm moves two circles apart if they are touching, and does not move circles that are isolated
# www.codeplastic.com/2017/09/09/controlled-circle-packing-with-processing

class ball:
    def __init__(self, u: float, v: float, radius: float, 
                angle: float = 0, p: float = 2):
        self.center = np.array([u, v])
        self.radius = radius
        self.angle = angle
        self.norm = p
        self.velocity = np.array([uniform(0,1), uniform(0,1)])
        self.acceleration = np.array([0,0])

    def apply_force(self, force):
        self.acceleration = np.add(self.acceleration, force)

    def update(self):
        self.velocity = np.add(self.velocity, self.acceleration)
        self.center = np.add(self.center, self.velocity)
        self.acceleration *= 0

class container:
    def __init__(self, radius, balls):
        self.iter = 0
        self.radius = radius
        self.balls = balls
        self.forces = [np.array([0,0])] * len(self.balls)
        self.nearballs = [0] * len(self.balls)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm

    def circle_distance(self, b1, b2): # this needs to change for p-balls
        res = np.linalg.norm(b1.center - b2.center) 
        return res
    
    def iterate(self):
        self.iter += 1
        for b in self.balls:
            self.checkContained(b)
            self.checkNeighbours(b)
            self.applyForce(b)

    def checkContained(self, b): # this will need to change for p-balls
        R = self.radius # radius of enclosing circle
        x = b.center[0]
        y = b.center[1]
        r = b.radius # radius of ball
        if (x**2 + y**2)**0.5 + r >= R: # if ball is outside container
            vr = self.normalize(b.velocity) * r

            px = x + vr[0]
            py = y + vr[1]
            p = np.array([px, py]) # point of contact between ball and container

            N = -1 * self.normalize(p) # normal vector 
            u = np.dot(b.velocity, N) * N
            v = np.subtract(b.velocity, u)

            b.velocity = np.subtract(v, u)

            b.update()
        
    def checkNeighbours(self, b):
        i = self.balls.index(b)
        for j in range(i+1, len(self.balls)):
            if self.circle_distance(b, self.balls[j]) < b.radius + self.balls[j].radius:
                return
        b.velocity[0] = 0
        b.velocity[1] = 0

    def getForce(self, b1, b2):
        force = np.array([0,0])

        d = self.circle_distance(b1, b2)

        if d > 0 and d < (b1.radius + b2.radius):
            diff = np.subtract(b1.center, b2.center)
            diff = self.normalize(diff)
            diff = np.divide(diff, d) 
            force = np.add(force, diff)
            # adding some randomness seems to make it do a little bit better?
            force = np.add(force, np.array([uniform(-1,1), uniform(-1,1)])) 
        
        return force

    def applyForce(self, b): # this may need to change depending if I want p-balls to rotate or not
        i = self.balls.index(b)
        for j in range(i+1, len(self.balls)):
            forceij = self.getForce(b, self.balls[j])

            if np.linalg.norm(forceij) > 0:
                self.forces[i] = np.add(self.forces[i], forceij)
                self.nearballs[i] += 1

                self.forces[j] = np.subtract(self.forces[j], forceij)
                self.nearballs[j] += 1

        if np.linalg.norm(self.forces[i]) > 0:
            self.forces[i] = np.subtract(self.forces[i], b.velocity)

        if self.nearballs[i] > 0:
            self.forces[i] = np.divide(self.forces[i], self.nearballs[i])

        force = self.forces[i]
        b.apply_force(force)
        b.update()

# create N circles with random centers and radii r contained (hopefully) inside a circle of radius R
def initialize_balls(N, R, r = 1):
    balls = []
    for i in range(N):
        u = 0.5*(R - 2*r)*uniform(-1, 1)
        v = 0.5*(R - 2*r)*uniform(-1, 1)
        b = ball(u, v, r)
        balls.append(b)
    return balls

# plotting purposes:

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

    if p != 2:
        # rotate to be parallel with horizontal
        R = [[np.cos(-theta), -np.sin(-theta)], 
            [np.sin(-theta), np.cos(-theta)]]
        xy = np.dot(R, [x, y])
        x = xy[0]
        y = xy[1]

    # compute distance to ball using p-norm
    return (np.abs(x - u)**p + np.abs(y - v)**p)**(1/p) - r

# draw each ball in the list (zero contour of signed_distance)
def plot_balls(balls, x_range = 5, y_range = 5):
    feature_x = np.arange(-x_range, x_range, 0.01)
    feature_y = np.arange(-y_range, y_range, 0.01)
    fig, ax = plt.subplots()
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = [0] * len(X)

    if isinstance(balls, ball):
        for k in range(len(X)):
                Z[k] = signed_distance(balls, X[k], Y[k])
        ax.contour(X, Y, Z, levels=[0], colors = 'black')
    else:
        for b in balls:
            for k in range(len(X)):
                Z[k] = signed_distance(b, X[k], Y[k])
            ax.contour(X, Y, Z, levels=[0], colors = 'black')

    plt.show()

# draw a ball on the plane (zero contour of signed distance)
def plot_ball(ball, ax, x_range = 5, y_range = 5):
    feature_x = np.arange(-x_range, x_range, 0.01)
    feature_y = np.arange(-y_range, y_range, 0.01)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = np.zeros_like(X)
    for k in range(len(X)):
        Z[k] = signed_distance(ball, X[k], Y[k])
    ax.contour(X, Y, Z, levels=[0], colors = 'black')

# draw each ball in list of N balls, and their containing ball
def draw(balls, R, ax):
    ax.set_aspect('equal') 
    ax.clear()
    plot_ball(ball(0, 0, R), ax, x_range = R, y_range = R)
    for i in range(len(balls)):
        plot_ball(balls[i], ax, x_range = R, y_range = R)

N = 10
R = 4
balls = initialize_balls(N, R)
con = container(R, balls)
frames = []

iter = 500
for i in range(iter):
    container.iterate(con)
    frames.append(deepcopy(con.balls))

fig, ax = plt.subplots()

def animate(i):
    balls = frames[i]
    draw(balls, R, ax)
    ax.set_title(f'N: {N:.2f}, Radius: {R:.2f}, Iteration: {i:.2f}')

# Create animation and save as GIF
anim = FuncAnimation(fig, animate, frames=len(frames), interval=200)
anim.save("bouncing.gif", writer=PillowWriter(fps=5))
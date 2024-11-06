import numpy as np
from scipy.optimize import minimize

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
    s = 2**0.5 # side length (unit 1-balls have side-length sqrt(2))
    
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
    # Check if projections overlap
    return max(proj1) >= min(proj2) and max(proj2) >= min(proj1)
    
# check rect1 = (x1, y1, theta1) and rect2 = (x2, y2, theta2) do not overlap:
# use seperating axis theorem to check all the normal planes, project onto those planes and show seperation in at least one
# if cannot find seperation then they intersect
def check_overlap(rect1, rect2):
    rect1 = get_vertices(rect1)
    rect2 = get_vertices(rect2)
    
    axes1 = get_axes(rect1)
    axes2 = get_axes(rect2)

    for axis in axes1 + axes2:
        proj1 = project(rect1, axis)
        proj2 = project(rect2, axis)
        
        if not overlap(proj1, proj2):
            return max(min(proj2) - max(proj1), min(proj1) - max(proj2))  # Separating axis found, rectangles do not overlap
    
    return -1  # No separating axis found, rectangles overlap

# check contained
def check_contained(rect, S):
    x, y, _ = rect  # Get the center coordinates
    distance_to_center = np.sqrt(x**2 + y**2)
    return S - distance_to_center - 1

def pack_squares(N):
    x0 = np.zeros(3*N + 1)
    S0 = N/2
    x0[-1] = S0

    def objective(x):
        return x[-1]
    
    constraints = []

    # non-overlapping constraints, i.e. d(ci, cj) >= 2
    for i in range(N):
        for j in range(1 + i, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: check_overlap(x[3*i:3*i + 3], x[3*j:3*j + 3])
            })

    # contained in enclosing circle, i.e. d(ci, 0) <= R - 1
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: check_contained(x[3*i:3*i + 3], x[-1])
        })

    result = minimize(objective, x0, constraints=constraints, method='SLSQP')

    if result.success:
        x_opt = result.x
        squares = x_opt[:3 * N].reshape((N, 3))
        S = x_opt[-1]
        return squares, S
    else:
        raise ValueError("Optimization failed!")
    
pack_squares(5)
import numpy as np
from numpy.linalg import inv

# np.set_printoptions(precision = 2, suppress = True)

s0 = 20 #screen coordinate 1
s1 = 25 #screen coordiante 2

r = 1280 # x value of frame
t = 720 # y value of frame

V = np.array([
    [r/2, 0, 0, r/2],
    [0, t/2, 0, t/2],
    [0, 0, 1/2, 1/2],
    [0, 0, 0, 1]
])

invV = inv(V)

far = -16/9
near = -9/16
right = 0
left = 0

P = np.array([
    [2*near/(right-1), 0, (right + left)/(right - left), 0],
    [0, 2*near/(top-bottom), 0, t/2],
    [0, 0, 1/2, 1/2],
    [0, 0, 0, 1]
])




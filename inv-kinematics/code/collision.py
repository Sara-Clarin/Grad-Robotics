import numpy as np

def normalize( matrix):

    #return matrix / np.linalg.norm(matrix, 'fro')
    return np.linalg.norm(matrix)

def line_sphere_intersection(p1, p2, c, r):
    """
    Implements the line-sphere intersection algorithm.
    https://en.wikipedia.org/wiki/Line-sphere_intersection

    :param p1: start of line segment --> o in the alg
    :param p2: end of line segment   --> x in the alg
    :param c: sphere center
    :param r: sphere radius
    :returns: discriminant (value under the square root) of the line-sphere
        intersection formula, as a np.float64 scalar
    """
    # FILL in your code here

    u = p2 - p1

    t1 = (u.dot(p1 - c) ) **2
    t2 = ((normalize(u))**2) * (
            (normalize(p1 - c) **2)  - r**2 
        )

    return np.float64(t1 - t2)

def test_disc(p1, p2, c, r):
    disc = line_sphere_intersection(p1, p2, c, r)
    print(f'{p1} , {p2}')

    if disc < 0: 
        print("Segment does not intersect")
    elif disc == 0:
        print("Segment intersects in exactly 1 point")
    else:
        print("Segment intersects in 2 points")
   
''' 
#Optional test code
r = 10
c = np.array([0,0])

disc = test_disc( np.array([11, -5]),np.array([11, 5]) , c, r)
disc = test_disc( np.array([11, 0]),np.array([11, 0]) , c, r)
disc = test_disc( np.array([-5, -10]),np.array([-5, 10]) , c, r)
'''

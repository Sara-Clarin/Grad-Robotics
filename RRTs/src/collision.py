from abc import ABC, abstractmethod

import numpy as np


class CollisionObject(ABC):
    """
    Abstract class for a parametrically defined collision object.
    """
    @abstractmethod
    def in_collision(self, target):
        """
        Checks whether target point is in collision. Points at the boundary of
        the object are in collision.

        :returns: Boolean indicating target is in collision.
        """
        pass


class CollisionBox(CollisionObject):
    """
    N-dimensional box collision object.
    """
    def __init__(self, location, half_lengths):
        """
        :params location: coordinates of the center
        :params half_lengths: half-lengths of the rectangle along each axis
        """
        self.location = np.asarray(location)
        self.half_lengths = np.asarray(half_lengths)
        self.ndim = self.location.shape[0]

    def in_collision(self, target):
        # FILL in your code here
        within = 0
        n_dims = len(target)
        #print("half lengths: ", self.half_lengths)

        for i in range( n_dims ):
            targ_dim = target[i]
            lower_bound = self.location[i] - self.half_lengths[i]
            upper_bound = self.location[i] + self.half_lengths[i]

            if targ_dim >= lower_bound and targ_dim <= upper_bound:
                within += 1

        #if within == (n_dims-1):
            #print("In collision!")
        return within == (n_dims )


class CollisionSphere(CollisionObject):
    """
    N-dimensional sphere collision object.
    """
    def __init__(self, location, radius):
        """
        :params location: coordinates of the center
        :params radius: radius of the circle
        """
        self.location = np.asarray(location)
        self.radius = radius

    def in_collision(self, target):
        # FILL in your code here
        #print("In collision with sphere!")
        #outside = 0
        #for i in range(len(target)):
        #    targ_dim = target[i]
        #    if 

        return np.linalg.norm(target - self.location) <= self.radius

'''
mySp = CollisionSphere( np.array([0,0,0]), 3)
myRect = CollisionBox( np.array([0,0,0]),np.array([3, 2, 1]))
point1 = np.array([1,1,1])

if mySp.in_collision(point1):
    print(point1)
    print("in collision")
else:
    print("not in collision")

point2 = np.array([10,1,3])

if mySp.in_collision(point2):
    print(point2)
    print("in collision")
else:
    print("not in collision")


myRect = CollisionBox( [0,0], [0.5, 0.5] )
points = [[0,0],[1, 0], [0, 0.5] ]

for point1 in points:
    print(point1)
    if myRect.in_collision(point1):
        print("in collision")
    else:
        print("not in collision")
    
    # expected: True False True
'''




import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from collision import CollisionBox, CollisionSphere
from rrt import RRT
import logging
import sys

#logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    print(len(sys.argv))
    print(sys.argv)
    
    loglevel = None

    if len(sys.argv) > 1:
        print("sara is right")
        loglevel = sys.argv[1]
    else:
        print("my suspicions were justified")

    if loglevel:
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)

    rrt = RRT(
        start_state=[0.2, 0.2],
        goal_state=[0.7, 0.7],
        dim_ranges=[(0, 1), (0, 1)])
    path = rrt.build()
    print('Path: ', path)

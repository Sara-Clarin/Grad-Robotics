import numpy as np

import logging

#logging.basicConfig(level=logging.DEBUG)

class RRT():
    """
    Simple implementation of Rapidly-Exploring Random Trees (RRT)
    """
    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def __eq__(self, other):
            if other is None or self is None:
                return False

            for i in range(len(self.state)):
                if self.state[i] != other.state[i]:
                    return False
            return True

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param statee: np.array of new child node's statee
            :returns: child Node object.
            """
            child = RRT.Node(state=state, parent=self)
            self.children.append(child)
            return child


    def __init__(self,
                 start_state,
                 goal_state,
                 dim_ranges,
                 obstacles=[],
                 step_size=0.05,
                 max_iter=1000):
        """
        :param start_state: Array-like representing the start state.
        :param goal_state: Array-like representing the goal state.
        :param dim_ranges: List of tuples representing the lower and upper
            bounds along each dimension of the search space.
        :param obstacles: List of CollisionObjects.
        :param step_size: Distance between nodes in the RRT.
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = RRT.Node(start_state, None)
        self.goal = RRT.Node(goal_state, None)
        self.dim_ranges = dim_ranges
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.penultimate = None
        self.path = []

        if (self.start.state.shape != self.goal.state.shape):
            raise AssertionError("Start and Goal states do not match dimension!")

    def build(self):
        """
        Build an RRT.

        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            # FILL in your code here

            logging.info('------------------')
            logging.debug(f'Iteration {k}')
            qrand = self._get_random_sample()

            qnear = self._get_nearest_neighbor(qrand) # closest neighbor of qrand in T
            new_node  = self._extend_sample(qrand, qnear)
            
            if new_node and self._check_for_completion(new_node):
                # FILL in your code here
                new_node.add_child( self.goal.state)

                self.penultimate = self.Node( new_node.state, new_node.parent) 
                self.penultimate.add_child( self.goal.state)

                path = self._trace_path_from_start()
                #path.append(self.start.state)
                #path.reverse()

                return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        # FILL in your code here
        rands = []
        for dim_boundary in self.dim_ranges:
            rands.append(
                np.random.uniform(dim_boundary[0], dim_boundary[1], 1)[0] )
        qrand = np.array( rands)
        return qrand

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        # FILL in your code here
        min_dist = 1000
        near_neighbor = self.start
         
        for neighbor in self.start:
            dist = np.linalg.norm(neighbor.state - sample)
            if dist < min_dist:
                logging.debug("New closest neighbor")
                min_dist = dist
                near_neighbor = neighbor

        # return np.argmin( [np.linalg.norm(x - sample) for x in neighbors])
        return near_neighbor

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        # FILL in your code here
        direction = (sample - neighbor.state) / np.linalg.norm(sample - neighbor.state)
        new_state = (direction* self.step_size) + neighbor.state
        if self._check_for_collision(  new_state):
            return None

        neighbor.add_child( new_state )
        return self.Node(new_state, neighbor)

    def _check_for_completion(self, node):
        """
        Check whether node is within self.step_size distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        # FILL in your code here
        distance = np.linalg.norm(self.goal.state - node.state )
        return distance <= self.step_size

    def _trace_path_from_start(self, node=None, step=0, found=False):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        # FILL in your code here

        # base cases
        if step == 0 and len(self.path) == 0:
            node = self.start

        elif node == self.penultimate:  # we redefined the base case here
            self.path.append(self.goal.state)
            return self.path
        elif node.children == []:
            return None

        # probably something recursive to see if it ends up at self.goal or not
        for child in node.children:

            sub_path =  self._trace_path_from_start(child, step+1)
            if sub_path:
                self.path.append(node.state)
                return self.path

            #if self.goal.state in sub_path:
                #self.path.append( node.state)

        return self.path

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision.
        """
        # FILL in your code here
        # Question: with the abstract class, how do we know if it's a sphere or not?

        for obstacle in self.obstacles:
            if obstacle.in_collision(sample):
                return True

        return False

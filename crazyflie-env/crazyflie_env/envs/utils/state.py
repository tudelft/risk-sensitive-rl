"""
State to describe state of your robot.
"""
import numpy as np

class FullState():
    def __init__(self, px, py, vx, vy, vf, radius, gx, gy, orientation, ranger_reflections):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.vf = vf
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.orientation = orientation

        self.position = np.array([self.px, self.py])
        self.velocity = np.array([self.vx, self.vy])
        self.goal_position = np.array([self.gx, self.gy])

        self.ranger_reflections = ranger_reflections

        self.state_tuple = (self.px, self.py, self.vf, self.radius, self.gx, self.gy, self.orientation,
                            self.ranger_reflections[0], self.ranger_reflections[1],
                            self.ranger_reflections[2], self.ranger_reflections[3])

    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])

    def __len__(self):
        return len(self.state_tuple)


class ObservableState():
    def __init__(self, position, velocity, goal_distance, orientation, ranger_reflections):
        self.position = position
        self.velocity = velocity
        self.goal_distance = goal_distance
        self.orientation = orientation
        self.ranger_reflections = ranger_reflections

        self.state_tuple = (self.position, self.goal_distance, self.orientation, *[ref for ref in self.ranger_reflections])
    
    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])

    def __len__(self):
        return len(self.state_tuple)

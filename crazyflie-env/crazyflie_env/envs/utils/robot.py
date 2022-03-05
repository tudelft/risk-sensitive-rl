import math
import logging
import numpy as np
import gym
from gym.utils import seeding
from crazyflie_env.envs.utils.action import ActionXY, ActionRotation
from crazyflie_env.envs.utils.state import FullState, ObservableState
from crazyflie_env.envs.utils.util import get_ranger_reflection

class Robot():
    def __init__(self, partial_observation=True):
        self.radius = 0.05
        self.v_pref = 1.0 # max possible velocity
        self.time_step = None
        self.fov = 2 * np.pi
        self.num_rangers = 4
        self.max_ranger_dist = 4.0
        #self.ranger_to_central = 0.012 # ranger to central distance
        self.partial_observation = partial_observation

        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.gx = None
        self.gy = None
        self.vf = None
        self.orientation = None
        self.ranger_reflections = None


    def set_state(self, px, py, vx, vy, gx, gy, vf, orientation, segments, radius=None):
        """Set initial state for the robot.
        Param: (position_x, position_y, goal_pos_x, goal_pos_y, vel_forward, theta_orientation, obstacle_segments, [optional] radius).
        Param Orientation: in rads
        """
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.vf = vf
        self.orientation = orientation
        self.ranger_reflections = self.get_ranger_reflections(segments)

        if radius is not None:
            self.radius = radius


    def get_ranger_reflections(self, segments):
        """
        set ranger_reflections according to obstacles in the environment
        add gaussian noise (mean=0, std=0.01) to each beam
        return np.array with shape (num_reflections,)
        """
        #TODO: add noise
        ranger_reflections = get_ranger_reflection(segments=segments, fov= self.fov, n_reflections=self.num_rangers, max_dist=self.max_ranger_dist,
                                    xytheta_robot=np.hstack((self.get_position(), self.orientation)))
        noise = np.random.normal(0.0, 0.01, ranger_reflections.shape)
        
        return np.clip(ranger_reflections + noise, 0.0, self.max_ranger_dist)


    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.vf, self.radius, self.gx, self.gy, self.orientation, self.ranger_reflections)


    def get_observable_state(self):
        return ObservableState(self.get_position(), self.get_velocity(), self.get_goal_distance(), self.orientation, self.ranger_reflections)


    def observe(self):
        if self.partial_observation:
            return self.get_observable_state()
        else:
            return self.get_full_state()


    def get_position(self):
        return np.array([self.px, self.py])


    def get_velocity(self):
        return np.array([self.vx, self.vy])


    def get_goal_position(self):
        return np.array([self.gx, self.gy])


    def compute_next_xy(self, action, dt):
        """
        compute next position according to forward velocity and orientation
        """
        assert isinstance(action, ActionXY)
        px = self.px + action.vx * dt
        py = self.py + action.vy * dt
        # TODO: add delay or noise into this function
        return np.array([px, py]) # implicitly return a tuple instead of two values


    def compute_next_orientation(self, action, dt):
        """
        action: ActionRotation
        rot > 0: ccw
        rot < 0: cw
        """
        orientation = (self.orientation + action.rot * dt) % (2 * np.pi)

        return orientation


    def compute_next_position(self, orientation, action, dt):
        """
        compute next position according to forward velocity and orientation
        """
        vx = action.vf * np.cos(orientation)
        vy = action.vf * np.sin(orientation)

        px = self.px + vx * dt
        py = self.py + vy * dt

        return np.array([px, py]) # implicitly return a tuple instead of two values


    def validate_action(self, action):
        assert isinstance(action, ActionRotation) or isinstance(action, ActionXY)


    def step(self, action, segments, next_orientation, next_position):
        """
        update orientation, position, and ranger reflection to next state.
        note that orientation and position won't be updated at same step.
        """
        self.validate_action(action)
        self.px, self.py = next_position[0], next_position[1]
        self.vf = action.vf
        self.orientation = next_orientation
        self.ranger_reflections = self.get_ranger_reflections(segments)
        
        #return self.get_observable_state()
        return self.observe()
    

    def step_xy(self, action, segments, next_position):
        """
        update orientation, position, and ranger reflection to next state.
        note that orientation and position won't be updated at same step.
        """
        self.px, self.py = next_position[0], next_position[1]
        self.vx, self.vy = action.vx, action.vy
        self.ranger_reflections = self.get_ranger_reflections(segments)
        # TODO: add vx, vy to full state space
        # self.vf makes no sense in this action space
        #return self.get_observable_state()
        return self.observe()


    def get_goal_distance(self):
        return np.linalg.norm(self.get_position() - self.get_goal_position())
    

    def reached_destination(self):
        return self.get_goal_distance() < self.radius
from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRotation = namedtuple('ActionRotation', ['vf', 'rot'])
import numpy as np
from crazyflie_env.envs.utils.obstacle import Obstacle

def point_to_segment_dist(end_point_1, end_point_2, point):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)
    """

    x1, y1 = end_point_1
    x2, y2 = end_point_2
    x3, y3 = point

    px = x2 - x1
    py = y2 - y1

    # if a line segment is a point
    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))
    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def plot_trajectory(waypoints):
    pass


def get_intersection(a1, a2, b1, b2):
    """
    param a1: np.array(x1, y1) line segment 1 - starting point
    param a2: np.array(x1', y1') line segment 1 - ending point
    param b1: np.array(x2, y2) line segment 2 - starting point
    param b2: np.array(x2', y2') line segment 2 - ending point
    return: np.array, point of intersection (if intersect); None (if not intersect)
    """

    def perp(a):
        # perpendicular vector
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    intersct = np.array((num / (denom.astype(float) + 1e-8)) * db + b1) # check divided by zero?

    delta = 1e-3
    condx_a = min(a1[0], a2[0]) - delta <= intersct[0] and max(a1[0], a2[0]) + delta >= intersct[0] # within a1_x--a2_x
    condy_a = min(a1[1], a2[1]) - delta <= intersct[1] and max(a1[1], a2[1]) + delta >= intersct[1] # within a1_y--a2_y
    condx_b = min(b1[0], b2[0]) - delta <= intersct[0] and max(b1[0], b2[0]) + delta >= intersct[0] # within b1_x--b2_x
    condy_b = min(b1[1], b2[1]) - delta <= intersct[1] and max(b1[1], b2[1]) + delta >= intersct[1] # within b1_y--b2_y

    if not (condx_a and condx_b and condy_a and condy_b):
        intersct = None

    return intersct


def get_ranger_reflection(segments, fov=2*np.pi, n_reflections=4, max_dist=4, xytheta_robot=np.array([0.0, 0.0, 0.0])):
    """
    param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
                    which is the return type of Obstacle().get_segments()
    param fov: sight range of the robot - for multiranger it will be 2 * pi
    param n_reflections: resolution = fov / n_reflections
    param max_dist: max distance the robot can see (/m). for multiranger, if no obstacle, 'ranger input distance' = max_dist
    param xy_robot: robot's position in the global coordinate system
    return: np.array with shape (n_reflections,), indicating the 'ranger input distance'
    """
    xy_robot = xytheta_robot[:2] # robot position
    theta_robot = xytheta_robot[2] # robot angle in rad

    ranger_angles = np.linspace(theta_robot, theta_robot + fov, num=n_reflections, endpoint=False)
    ranger_reflections = max_dist * np.ones(n_reflections) # initialize each ranger reflection to its max possible distance

    for segment in segments:
        xy_start, xy_end = np.array(segment[:2]), np.array(segment[2:]) # start and end points of each segment
        for i, angle in enumerate(ranger_angles):
            max_xy_ranger = xy_robot + np.array([max_dist * np.cos(angle), max_dist * np.sin(angle)])
            intersection = get_intersection(xy_start, xy_end, xy_robot, max_xy_ranger)
            if intersection is not None:
                radius = np.sqrt(np.sum((intersection - xy_robot) ** 2))
                if radius < ranger_reflections[i]:
                    ranger_reflections[i] = radius

    return ranger_reflections


def load_obstacle_config(environment):
    # TODO: (can be randomly) generate a desired env with obstacles
    pass
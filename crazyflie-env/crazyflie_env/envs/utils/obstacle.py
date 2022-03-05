import numpy as np

class Obstacle():
    """
    Static rectangular obstacle, given the centroid and dimension of the obstacle,
    Return the segment points. E.g. Boxes, walls.
    TODO: Cylinders.
    """
    def __init__(self, centroid, wx, wy, angle=0):
        """
        param centroid: (tuple or list) centroid of the obstacle
        param wx: width of the obstacle (>=0)
        param wy: height of the obstacle (>=0)
        param angle: anti-clockwise rotation from the x-axis
        """
        self.centroid = centroid
        self.wx = wx
        self.wy = wy
        self.angle = angle

        wx_cos = self.wx * np.cos(self.angle)
        wx_sin = self.wx * np.sin(self.angle)
        wy_cos = self.wy * np.cos(self.angle)
        wy_sin = self.wy * np.sin(self.angle)

        self.BR_x = self.centroid[0] + 0.5 * (wx_cos + wy_sin) # BR bottom-right
        self.BR_y = self.centroid[1] + 0.5 * (wx_sin - wy_cos)
        self.BL_x = self.centroid[0] - 0.5 * (wx_cos - wy_sin)
        self.BL_y = self.centroid[1] - 0.5 * (wx_sin + wy_cos)
        self.TL_x = self.centroid[0] - 0.5 * (wx_cos + wy_sin)
        self.TL_y = self.centroid[1] - 0.5 * (wx_sin - wy_cos)
        self.TR_x = self.centroid[0] + 0.5 * (wx_cos - wy_sin)
        self.TR_y = self.centroid[1] + 0.5 * (wx_sin + wy_cos)
    
    def bl_anchor_point(self):
        """
        return anchor point (xy) for matplot
            +------------------+
            |                  |
            height             |
            |                  |
           (xy)---- width -----+
        """
        return self.BL_x, self.BL_y


    def get_segments(self):
        """
        return: A wall ((x1, y1, x1', y1'))
                Or a box: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        seg_bottom = (self.BL_x, self.BL_y, self.BR_x, self.BR_y)
        seg_left = (self.BL_x, self.BL_y, self.TL_x, self.TL_y)

        if self.wy == 0: # if no height
            return (seg_bottom,)
        elif self.wx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (self.TL_x, self.TL_y, self.TR_x, self.TR_y)
            seg_right = (self.BR_x, self.BR_y, self.TR_x, self.TR_y)
            return (seg_bottom, seg_top, seg_left, seg_right)
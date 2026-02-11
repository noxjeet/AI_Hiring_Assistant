import cv2
import numpy as np

class AlignmentLogic: # The class is defined here, so you don't import it.
    def __init__(self):
        self.matrix = None

    def set_calibration(self, rgb_pts, thermal_pts):
        self.matrix = cv2.getAffineTransform(
            np.float32(rgb_pts), 
            np.float32(thermal_pts)
        )

    def map_points(self, points):
        if self.matrix is None:
            return points
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_pts = self.matrix.dot(points_ones.T).T
        return transformed_pts.astype(int)
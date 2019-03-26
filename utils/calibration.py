import cv2
import numpy as np


def FitHomography(from_pts, to_pts):
    from_pts = np.float32(from_pts)
    from_pts = np.array(from_pts)
    from_pts = np.float32(from_pts)
    to_pts = np.float32(to_pts)
    to_pts = np.array(to_pts)
    to_pts = np.float32(to_pts)

    M, status = cv2.findHomography(from_pts, to_pts, method=0)
    M_inv = np.linalg.inv(M)

    class Transformation:
        def getMatrix(self):
            return M

        def Transform(self, pt):
            pts = np.array([[pt]])
            pts = np.float32(pts)
            return cv2.perspectiveTransform(pts, M)[0][0]

        def Transform_inv(self, pt):
            pts = np.array([[pt]])
            pts = np.float32(pts)
            return cv2.perspectiveTransform(pts, M_inv)[0][0]

    return Transformation()

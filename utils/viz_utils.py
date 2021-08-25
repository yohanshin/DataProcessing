import numpy as np
import cv2
import matplotlib.pyplot as plt


"""Utility functions and constants for visualizing data"""

__author__ = "Soyong SHin"


CONNECTIVITY_DICT = {
        "op26": [(0, 9), (0, 3), (9, 10), (10, 11), (3, 4), (4, 5),
                 (2, 12), (2, 6), (12, 13), (13, 14), (6, 7), (7, 8),
                 (0, 1), (0, 2), 
                 (1, 17), (17, 18), (1, 15), (15, 16),
                 (24, 14), (14, 22), (22, 23), 
                 (21, 8), (8, 19), (19, 20)]}

COLOR_DICT = {
        "op26": [(255, 85, 0), (170, 255, 0), (255, 170, 0), (255, 170, 0), (85, 255, 0), (0, 255, 0), # arms and shoulder
                 (0, 255, 85), (0, 170, 255), (0, 255, 170), (0, 255, 255), (0, 85, 255), (0, 0, 255), # legs and hip
                 (255, 0, 85), (255, 0, 0), # spine and neck
                 (255, 0, 170), (255, 0, 255), (170, 0, 255), (85, 0, 255), # face
                 (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255) # feet
                 ]}
                 
COLOR_DICT_KEYPOINTS = {
        "op26": [(255, 0, 0), (255, 0, 170), (255, 85, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 170, 255), (0, 85, 255), (0, 0, 255),
                 (255, 85, 0), (255, 170, 0), (255, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (170, 0, 255), (85, 0, 255), (255, 0, 170), (170, 0, 255),
                 (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (255, 0, 0)]}

def project2D(x3d, imw, imh, K, R, t, dist=None):
    """
    x3d : F * N * 4 numpy array
    K : 3 * 3 numpy array
    R : 3 * 3 numpy array
    t : 3 * 1 numpy array
    dist : 5 * 1 numpy array
    """

    conf_mask = x3d[:, :, -1] > 1e-5
    x3d = x3d[:, :, :-1]

    x2d = np.zeros_like(x3d[:, :, :2])
    R2_criterion = np.zeros_like(x3d[:, :, 0])

    for J in range(x3d.shape[1]):
        """ J is joint index """
        x = np.dot(R, x3d[:, J].T) + t
        xp = x[:2] / x[2]

        if dist is not None:
            X2 = xp[0] * xp[0]
            Y2 = xp[1] * xp[1]
            XY = X2 * Y2
            R2 = X2 + Y2
            R4 = R2 * R2
            R6 = R4 * R2
            R2_criterion[:, J] = R2

            radial = 1.0 + dist[0] * R2 + dist[1] * R4 + dist[4] * R6
            tan_x = 2.0 * dist[2] * XY + dist[3] * (R2 + 2.0 * X2)
            tan_y = 2.0 * dist[3] * XY + dist[2] * (R2 + 2.0 * Y2)

            xp[0, :] = radial * xp[0, :] + tan_x
            xp[1, :] = radial * xp[1, :] + tan_y

        pt = np.dot(K[:2, :2], xp) + K[:2, 2:]
        x2d[:, J, :] = pt.T

    x2d = x2d.astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < imw)
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < imh)
    visible = np.logical_and(x_visible, y_visible)
    vis_mask = np.logical_and(visible, R2_criterion < 1.)
    mask = np.logical_and(conf_mask, vis_mask)

    return x2d, mask

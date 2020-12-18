import numpy as np
import cv2
import matplotlib.pyplot as plt


"""Utility functions and constants for visualizing data"""

__author__ = "Soyong SHin"

CONNECTIVITY_DICT = {
    "op26": [(18, 17), (17, 1), (1, 15), (15, 16),  # face
             (0, 1), (0, 2),    # body
             (5, 4), (4, 3), (3, 0),    # left arm
             (11, 10), (10, 9), (9, 0),     # right arm
             (8, 7), (7, 6), (6, 2),    # left leg
             (14, 13), (13, 12), (12, 2),   # right leg
             (19, 20), (20, 8), (8, 21),    # left foot
             (22, 23), (23, 14), (14, 24)]  # right foot
}

COLOR_DICT = {
    "op26": [
        (220,20,60), (220,20,60), (220,20,60), (220,20,60),  # face
        (153, 0, 0), (153, 0, 0),  # body
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # left arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0),   # right arm
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # left leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # right leg
        (0,0,205), (0,0,205), (0,0,205),    # left foot
        (60,179,113), (60,179,113), (60,179,113)    # right foot
    ]
}
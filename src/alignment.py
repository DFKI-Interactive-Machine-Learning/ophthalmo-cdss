# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import cv2 as cv
import numpy as np


def compute_matches(oct1, oct2):
    """Computes the matches between the two images using the ORB algorithm.
        Returns the matches, keypoints of the first image and keypoints of the second image sorted by distance."""
    # Use a feature matching technique ORB (Oriented FAST and Rotated BRIEF) to
    # find key points and descriptors in both images
    orb = cv.ORB_create()
    # Find the key points and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(oct1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(oct2, None)
    # Create a Brute Force Matcher to match the descriptors:
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match the descriptors
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance), keypoints1, keypoints2


def compute_transformation(oct1, oct2, use_shearing=True, n_matches=100):
    """Computes a euclidian transformation matrix (only rotation and translation) for matching oct1 with oct2."""

    # Compute the matches between the two images
    matches, keypoints1, keypoints2 = compute_matches(oct1, oct2)

    # Choose the top 50 matches and store their keypoints
    good_matches = matches[:n_matches]

    # Get the keypoint coordinates for the matches
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])

    # Find a rotation and translation matrix
    if use_shearing:
        return cv.estimateAffine2D(points2, points1)[0]
    else:
        return cv.estimateAffinePartial2D(points2, points1)[0]


def align_slices(slice_1, slice_2, transform_matrix):
    """ Aligns oct2 on key points of oct1."""
    return cv.warpAffine(slice_2, transform_matrix, (slice_1.shape[1], slice_1.shape[0]))

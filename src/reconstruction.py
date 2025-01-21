# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import numpy as np
import cv2 as cv
import scipy.spatial as ss
from src.util import timing
from typing import List, Dict


def get_XYZ_of_surface(grid, oh_masks, downsample=10):
    """Computes XYZ points of the top pixels of the masks. Basically returns points that are recovered by going through
        each column of the mask from the top and return the first pixel that is true. Downsampling is used to reduce
        the number of points to be plotted, hence increases performance. """
    X = []
    Y = []
    Z = []
    for i, mask in enumerate(oh_masks):
        rows, cols = np.nonzero(mask)
        unique_cols = np.unique(cols)
        unique_cols = unique_cols[::downsample]
        for col in unique_cols:
            first_index = np.nonzero(cols == col)[0][-1]
            Z.append(rows[first_index])
            X.append(col)
            Y.append(grid[i][1])
    return X, Y, Z


class Reconstruction:
    """A class for the reconstruction of the masks. The class contains the convex hull of the points. The class
    provides methods to compute the average thickness, volume, surface area and surface map of the reconstruction."""
    def __init__(self, points, label):
        self.label = label
        self.hull = ss.ConvexHull(points, qhull_options="QJ")

    def get_average_thickness(self):
        """Computes the average thickness of the reconstruction."""
        return self.hull.volume / self.hull.area

    def get_volume(self):
        """Computes the volume of the reconstruction."""
        return self.hull.volume

    def get_surface_area(self):
        """Computes the area of the reconstruction."""
        return self.hull.area

    def get_surface_map(self, map_type="top", downsample_factor=.1):
        """Computes the surface of the reconstruction. The surface is defined as the top or bottom surface of the
        reconstruction. The surface is computed by going through each column of the reconstruction and returning the
        first point that is true. Downsampling is used to reduce the number of points to be plotted, hence increases
        performance.
        :param map_type: The type of map to return. Can be "top", "bottom" or "thickness". If "thickness" is chosen,
        the thickness of the reconstruction is returned. If "top" or "bottom" is chosen, the top or bottom surface
        is returned.
        :param downsample_factor: The factor by which the number of points is reduced.
        :return: A tuple of X, Y, Z coordinates of the surface.
        """
        X = []
        Y = []
        Z = []
        for i, y in enumerate(np.unique(self.y)):
            mask = self.points[:, 1] == y
            points_2d = self.points[mask][:, [0, 2]]
            unique_cols = np.unique(points_2d[:, 0])
            sample_rate = max(int(len(unique_cols) * downsample_factor), 1)
            unique_cols = unique_cols[::sample_rate]
            for col in unique_cols:
                if map_type == "top":
                    z_value = np.max(points_2d[points_2d[:, 0] == col][:, 1])  # Top surface
                elif map_type == "bottom":
                    z_value = np.min(points_2d[points_2d[:, 0] == col][:, 1])  # Bottom surface
                elif map_type == "thickness":
                    z_value = (np.max(points_2d[points_2d[:, 0] == col][:, 1]) -
                               np.min(points_2d[points_2d[:, 0] == col][:, 1]))  # Thickness
                else:
                    raise ValueError("Invalid type.")
                Z.append(z_value)
                X.append(col)
                Y.append(y)
        return X, Y, Z

    def get_top_down_contour(self):
        """Computes the top down view of the reconstruction."""
        X = self.x
        Y = self.y
        convex_hull_2d = ss.ConvexHull(np.stack([X, Y], axis=1), qhull_options="QJ")
        points = convex_hull_2d.points[convex_hull_2d.vertices]
        contour = np.expand_dims(points, 1).astype(np.int32)
        assert len(contour.shape) == 3
        return contour

    @property
    def points(self):
        return self.hull.points[self.hull.vertices]

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def z(self):
        return self.points[:, 2]


class ReconstructionBuilder:
    def __init__(self, contours, label):
        if contours.shape[1] == 1:
            contours = contours.squeeze(1)
        self.contours = contours
        self.label = label

    def add(self, contours):
        if contours.shape[1] == 1:
            contours = contours.squeeze(1)
        self.contours = np.concatenate([self.contours, contours], axis=0)

    def contour_distance(self, contour):
        distances = []
        for point in contour:
            distances.append(np.linalg.norm(self.contours - point, axis=1))
        return np.array(distances)

    def y_distance(self, y):
        return np.abs(self.contours[:, 1] - y)

    def smallest_contour_distance(self, contour):
        return np.min(self.contour_distance(contour))

    def smallest_y_distance(self, y):
        return np.min(self.y_distance(y))

    def is_in_distance(self, centroid, distance):
        return np.any(self.contour_distance(centroid) <= distance)

    def get_contours(self):
        return self.contours

    def get_n_points(self):
        return self.contours.shape[0]

    def lies_in_plane(self):
        return len(np.unique(self.contours[:, 1])) == 1

    def offset_centroid_in_dimension_as_new_point(self, dim, value):
        centroid = np.mean(self.contours, axis=0)
        new_contours = centroid.copy()
        new_contours[dim] += value
        self.add(np.expand_dims(new_contours, axis=0))

    def build(self, interpolation_value):
        """Finalizes the Reconstruction. If the object has too few points it adds artificial points."""
        if self.get_n_points() < 3:
            self.offset_centroid_in_dimension_as_new_point(0, -5)
            self.offset_centroid_in_dimension_as_new_point(0, 5)
            self.offset_centroid_in_dimension_as_new_point(2, -5)
            self.offset_centroid_in_dimension_as_new_point(2, 5)
        if self.lies_in_plane():
            self.offset_centroid_in_dimension_as_new_point(1, -interpolation_value)
            self.offset_centroid_in_dimension_as_new_point(1, interpolation_value)
        if self.get_n_points() < 4:
            self.offset_centroid_in_dimension_as_new_point(0, -6)
            self.offset_centroid_in_dimension_as_new_point(0, 6)
            self.offset_centroid_in_dimension_as_new_point(2, -6)
            self.offset_centroid_in_dimension_as_new_point(2, 6)
        return Reconstruction(self.contours, self.label)


@timing
def reconstruct_objects(vol, labels: List[int], max_allowed_distance=50) -> Dict[int, List[Reconstruction]]:
    """Reconstructs the objects from the masks.
    :param vol: A VOL object.
    :param labels: A list of labels to reconstruct.
    :param max_allowed_distance: The maximum distance between two contours to be considered as the same object. Default
            is 20 mikrometer.
    :return: A list of objects. Each object is a list of contours."""
    # Create a dictionary for each label. The dictionary contains a list of reconstructions as values.
    finished_reconstructions = {label: [] for label in labels}
    wip_reconstructions = {label: [] for label in labels}
    # Iterate over all masks.
    for i, mask in enumerate(vol.masks):
        y = vol.grid[i][1]
        slice_distance = abs(y - vol.grid[i - 1][1]) if i != 0 else y  # The distance between slices
        # Iterate over all labels in the mask.
        for label in labels:
            label_mask = np.array(mask == label)
            if np.any(label_mask):
                # If the label is present in the mask, then compute the contours.
                binary_fluid_mask = (label_mask * 255).astype(np.uint8)
                contours, _ = cv.findContours(binary_fluid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
                if wip_reconstructions[label]:
                    # If there are already reconstructions for the label, then compute the distances between the
                    # contours and the reconstructions.
                    distances = []
                    for rec in wip_reconstructions[label]:
                        rec_distances = []
                        for contour in contours:
                            contour_3d = np.stack([contour[:, 0, 0], [y] * len(contour), contour[:, 0, 1]], axis=1)
                            rec_distances.append(rec.smallest_contour_distance(contour_3d))
                        distances.append(rec_distances)
                    distances = np.array(distances)
                    # Iterate starting from the contours with the smallest distances and
                    # add the contours to the corresponding reconstruction.
                    for j, contour in sorted(list(enumerate(contours)), key=lambda x: np.min(distances[:, x[0]])):
                        contour_3d = np.stack([contour[:, 0, 0], [y] * len(contour), contour[:, 0, 1]], axis=1)
                        smallest_distance, index = np.min(distances[:, j]), np.argmin(distances[:, j])
                        if smallest_distance <= max_allowed_distance:
                            wip_reconstructions[label][index].add(contour_3d)
                        else:
                            wip_reconstructions[label].append(ReconstructionBuilder(contour_3d, label))
                else:
                    # If there are no reconstructions for the label, then create a new reconstruction for each contour.
                    for contour in contours:
                        contour_3d = np.stack([contour[:, 0, 0], [y] * len(contour), contour[:, 0, 1]], axis=1)
                        wip_reconstructions[label].append(ReconstructionBuilder(contour_3d, label))
                for j, reconstruction in reversed(list(enumerate(wip_reconstructions[label]))):
                    if reconstruction.smallest_y_distance(y) > min(slice_distance, max_allowed_distance):
                        # If the reconstruction is too far away from the current slice or the maximum allowed distance,
                        # then finalize it and add it to the finished objects list.
                        finished_reconstructions[label].append(reconstruction.build(
                            min(slice_distance / 2.0, max_allowed_distance)
                            )
                        )
                        # Remove the reconstruction from the wip_reconstructions list.
                        wip_reconstructions[label].pop(j)
    # Finally, add all remaining reconstructions to the finished objects list.
    for label in labels:
        for rec in wip_reconstructions[label]:
            finished_reconstructions[label].append(rec.build(min(slice_distance / 2.0, max_allowed_distance)))
    objects = finished_reconstructions
    return objects


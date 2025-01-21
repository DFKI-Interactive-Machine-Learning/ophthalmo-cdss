# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import numpy as np
import cv2 as cv
import torch
from logging import getLogger

import config
from config import *
import torch.nn as nn

logger = getLogger(__name__)


class SegmentationModel(nn.Module):
    """ Base class for segmentation models. If you want to implement a new segmentation model, inherit from this class
        and implement the segment_slice and segment_stack methods. """
    classes: dict[int, str] = {
            0: "Background",
            1: "IPL",
            2: "OPL",
            3: "ELM",
            4: "EZ",
            5: "RPE",
            6: "BM",
            7: "Choroidea",
            8: "Drusen",
            9: "PED",
            10: "Fluid",
        }

    def __init__(self, classes: dict[int, str] = None, in_size=(256, 256), *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent class constructor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, you can manually set this
        self.model = None  # The model should be initialized in the child class
        self.in_size = in_size  # The input size of the model
        if classes is not None:
            self.classes = classes  # The classes of the model as a dictionary mapping from class index to class name
        self.num_classes = len(self.classes)

    def segment_slice(self, oct_slice: np.ndarray) -> np.ndarray:
        """Segment a single OCT slice.
            :param oct_slice: A 2D numpy array of shape (height, width) containing the OCT slice.
            :return: A 2D numpy array of shape (height, width) containing the segmentation mask. Entries are class
                indices
                as defined in the classes attribute.
        """
        raise NotImplementedError("The base class SegmentationModel does not implement the segment_slice method.")

    def segment_stack(self, oct_stack) -> np.ndarray:
        """Segment a stack of OCT slices.
            :param oct_stack: A 3D numpy array of shape (n_slices, height, width) containing the OCT stack.
            :return: A 3D numpy array of shape (n_slices, height, width) containing the segmentation mask. Entries are
                class indices as defined in the classes attribute.
        """
        raise NotImplementedError("The base class SegmentationModel does not implement the segment_stack method.")

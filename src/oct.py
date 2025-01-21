# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.


import array
import codecs
import datetime
import io
import struct
from collections import OrderedDict

import config
from src.segmentation import *
from src.reconstruction import reconstruct_objects
from config import LAYERS_TO_COLOR, segmentation_model
from logging import getLogger

import cv2 as cv
import numpy as np

logger = getLogger(__name__)


class OCT:
    @property
    def oct(self):
        """
        Retrieve OCT volume as a 3D numpy array.

        Returns:
            3D numpy array with OCT intensities as 'uint8' array

        """
        pass

    @property
    def masks(self):
        """
        Retrieve predicted masks as 3D numpy array. Each entry in the array represents the layer for that pixel.
            E.g.: If that entry has value 8 then that pixel belongs to class 8.
        :return: 3D numpy array containing class identifiers
        """
        pass

    def get_volume_of_reconstructed_objects(self, label):
        """Get the volume of reconstructed objects for a given label."""
        pass

    def get_number_of_reconstructed_objects(self, label):
        """Get the number of reconstructed objects for a given label."""
        pass

    def get_thickness_map(self, label: int):
        """Get the thickness map for a given label. If no label is given,
            the total retinal thickness map is computed."""
        pass

    def get_mean_thickness(self, label: int):
        """Get the mean thickness of a given label. If no label is given, the total retinal thickness is computed."""
        pass

    def get_as_rgb_mask(self, index):
        """
        Get a mask slice as an RGB mask.
        :return: 3D numpy array containing RGB values
        """
        pass

    def get_IRSLO_segmentation(self, labels=None, highlight_slice=None):
        """Get IR SLO image with segmentation overlay."""
        pass

    @property
    def irslo(self):
        """
        Retrieve IR SLO image as 2D numpy array

        Returns:
            2D numpy array with IR reflectance SLO image as 'uint8' array.

        """
        pass

    @property
    def grid(self):
        """
        Retrieve the IR SLO pixel coordinates for the B scan OCT slices

        Returns:
            2D numpy array with the number of b scan images in the first dimension
            and x_0, y_0, x_1, y_1 defining the line of the B scan on the pixel
            coordinates of the IR SLO image.

        """
        pass

    def is_followup(self, reference_id):
        """Returns True if the file is a follow-up of the reference_id."""
        pass

    def renderIRslo(self, filename, renderGrid=False, highlight_slice=None):
        """
        Renders IR SLO image as a PNG file and optionally overlays grid of B scans

        Args:
            filename (str): filename to save IR SLO image
            renderGrid (bool): True will render red lines for the location of the B scans.

        Returns:
            None

        """
        pass

    def renderOCTscans(self, filepre="oct", renderSeg=False):
        """
        Renders OCT images a PNG file and optionally overlays segmentation lines

        Args:
            filepre (str): filename prefix. OCT Images will be named as "<prefix>-001.png"
            renderSeg (bool): True will render colored lines for the segmentation of the RPE, ILM, and NFL on the B scans.

        Returns:
            None

        """
        pass


    @property
    def fileHeader(self):
        """
        Retrieve vol header fields

        Returns:
            Dictionary with the following keys
                - version: version number of vol file definition
                - numBscan: number of B scan images in the volume
                - octSizeX: number of pixels in the width of the OCT B scan
                - octSizeZ: number of pixels in the height of the OCT B scan
                - distance: unknown
                - scaleX: resolution scaling factor of the width of the OCT B scan
                - scaleZ: resolution scaling factor of the height of the OCT B scan
                - sizeXSlo: number of pixels in the width of the IR SLO image
                - sizeYSlo: number of pixels in the height of the IR SLO image
                - scaleXSlo: resolution scaling factor of the width of the IR SLO image
                - scaleYSlo: resolution scaling factor of the height of the IR SLO image
                - fieldSizeSlo: field of view (FOV) of the retina in degrees
                - scanFocus: unknown
                - scanPos: Left or Right eye scanned
                - examTime: Datetime of the scan (needs to be checked)
                - scanPattern: unknown
                - BscanHdrSize: size of B scan header in bytes
                - ID: unknown
                - ReferenceID
                - PID: unknown
                - PatientID: Patient ID string
                - DOB: Date of birth
                - VID: unknown
                - VisitID: Visit ID string
                - VisitDate: Datetime of visit (needs to be checked)
                - GridType: unknown
                - GridOffset: unknown

        """
        pass

    def bScanHeader(self, slicei):
        """
        Retrieve the B Scan header information per slice.

        Args:
            slicei (int): index of B scan

        Returns:
            Dictionary with the following keys
                - startX: x-coordinate for B scan on IR. (see getGrid)
                - startY: y-coordinate for B scan on IR. (see getGrid)
                - endX: x-coordinate for B scan on IR. (see getGrid)
                - endY: y-coordinate for B scan on IR. (see getGrid)
                - numSeg: 2 or 3 segmentation lines for the B scan
                - quality: OCT signal quality
                - shift: unknown

        """
        pass

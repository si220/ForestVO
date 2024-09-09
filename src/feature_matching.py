"""
Perform LightGlue feature matching on image pairs
Taken from demo script on LightGlue repo

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import torch
import sys
import os
# LightGlue imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import rbd

class FeatureMatching():
    def __init__(self, device) -> None:
        # initialise SuperPoint and LightGlue
        self.feature_detector = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.feature_matcher = LightGlue(features="superpoint").eval().to(device)

        self.device = device

    def match_img_pair(self, img_0, img_1) -> torch.Tensor:
        """
        obtain coordinates of matched keypoints across image pair
        which are then passed into the pose estimation transformer

        This function extracts SuperPoint features from an image pair,
        matches features across the 2 images using LightGlue and then
        returns the coordinates of the matched kpts

        Parameters
        ----------
        self.img_0 : torch.Tensor
            first image
            shape (3,H,W) normalised in range [0,1]
        self.img_1 : torch.Tensor
            second image
            shape (3,H,W) normalised in range [0,1]

        Returns
        -------
        kpts_coords : torch.Tensor
            2D coordinates of matched kpts found in both images
            shape (N,4)
        """
        # disable gradient computation for faster inference
        with torch.no_grad():
            # SuperPoint feature detection
            feats0 = self.feature_detector.extract(img_0.to(self.device))
            feats1 = self.feature_detector.extract(img_1.to(self.device))

            # LightGlue feature matching
            matches01 = self.feature_matcher({"image0": feats0, "image1": feats1})

            # remove batch dimension
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # get keypoints and matches
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

            # get 2D coordinates of matched keypoints in both images
            points0 = feats0['keypoints'][matches[..., 0]]  # 2D coordinates in first image, shape (N,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # 2D coordinates in second image, shape (N,2)

            # concatenate the 2D coordinates into a single tensor of shape (N, 4)
            kpts_coords = torch.cat([points0, points1], dim=1)

        return kpts_coords
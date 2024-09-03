"""
Pose Estimation Dataset

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from LightGlue.lightglue.utils import load_image

class PoseEstimationDataset(Dataset):
    def __init__(self, data_seq, feature_matcher, pose_transforms, device) -> None:
        """
        create list of data sequences from txt file

        Parameters
        ----------
        data_seq : str
            path to txt file containing list of which data sequences to use
        feature_matcher : class
            instance of FeatureMatching() class
        pose_transforms : class
            instance of PoseTransforms() class
        device : torch.device
            use GPU if available otherwise CPU
        """
        self.feature_matcher = feature_matcher
        self.pose_transforms = pose_transforms
        self.device = device
        self.samples = []
        
        with open(data_seq, 'r') as file:
            paths = file.readlines()

        # create a list of sample paths without loading the data
        for path in paths:
            path = path.strip()

            if 'left' in path:
                image_dir = os.path.join(path, 'image_left/')
                pose_file = os.path.join(path, 'pose_left.txt')
            else:
                image_dir = os.path.join(path, 'image_right/')
                pose_file = os.path.join(path, 'pose_right.txt')

            image_files = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
            
            for i in range(len(image_files) - 1):
                self.samples.append({
                    'img0_path': os.path.join(image_dir, image_files[i]),
                    'img1_path': os.path.join(image_dir, image_files[i + 1]),
                    'pose_file': pose_file,
                    'pose_indices': (i, i+1)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This function parses images and gt poses,
        feature matching is performed on image pairs
        returns 2D coordinates of matched features, gt relative translation, gt relative rotation

        Parameters
        ----------
        idx : int
            index of data sample

        Returns
        -------
        kpts_coords, rel_translation, rel_rot_six_d : torch.Tensor, torch.Tensor, torch.Tensor
            kpts_coords are the 2D coordinates of matched kpts found in both images
            rel_translation is the relative 3D camera translation given as [x,y,z]
            rel_rot_six_d is the relative 6D camera rotation given as the first 2 cols of a 3x3 rotation matrix
            shapes (N,4), (1,3), (1,6)
        """
        sample = self.samples[idx]
    
        # use LightGlue load_image function to load images as torch.Tensors
        img_0 = load_image(sample['img0_path'])
        img_1 = load_image(sample['img1_path'])

        # load poses
        poses = np.loadtxt(sample['pose_file'])
        i, j = sample['pose_indices']
        pose_0, pose_1 = poses[i], poses[j]

        # get coordinates of matched keypoints
        kpts_coords = self.feature_matcher.match_img_pair(img_0, img_1)

        # convert poses to transformation matrices
        transformation_mat_0 = self.pose_transforms.pose_to_mat(pose_0[:3], pose_0[3:])
        transformation_mat_1 = self.pose_transforms.pose_to_mat(pose_1[:3], pose_1[3:])

        # get relative pose
        rel_translation, rel_rot_six_d = self.pose_transforms.get_relative_pose(transformation_mat_0, transformation_mat_1)

        # convert rel_translation from np array to torch.Tensor of type float32
        rel_translation = torch.from_numpy(rel_translation).to(self.device).float()
        # convert rel_rot_six_d from list to torch.Tensor of type float32
        rel_rot_six_d = torch.tensor(rel_rot_six_d).to(self.device).float()

        return kpts_coords.float(), rel_translation, rel_rot_six_d
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
    def __init__(self, data_seq=None, feature_matcher=None, pose_transforms=None, device=None, preprocessed_dir=None, preprocess=False):
        """
        Initialise the dataset either by preprocessing or by loading preprocessed data

        Parameters
        ----------
        data_seq : str, optional
            Path to txt file containing list of which data sequences to use. Required if preprocess=True
        feature_matcher : class, optional
            Instance of FeatureMatching() class. Required if preprocess=True
        pose_transforms : class, optional
            Instance of PoseTransforms() class. Required if preprocess=True
        device : torch.device, optional
            Use GPU if available otherwise CPU. Required if preprocess=True
        preprocessed_dir : str, optional
            Path to directory containing preprocessed data. Required if preprocess=False
        preprocess : bool, optional
            Whether to preprocess the dataset and save it, or load preprocessed data. Default is False
        """
        if preprocess:
            if not (data_seq and feature_matcher and pose_transforms and device):
                raise ValueError("data_seq, feature_matcher, pose_transforms, and device are required for preprocessing")
            
            self.preprocessed_dir = preprocessed_dir
            self.feature_matcher = feature_matcher
            self.pose_transforms = pose_transforms
            self.device = device
            self.samples = []
            self.preprocess_and_save(data_seq)

        else:
            if not preprocessed_dir:
                raise ValueError("preprocessed_dir is required for loading preprocessed data")
            
            self.samples = [os.path.join(preprocessed_dir, file) for file in sorted(os.listdir(preprocessed_dir))]

    def preprocess_and_save(self, data_seq):
        """
        Preprocess the dataset and save the preprocessed data to disk

        Parameters
        ----------
        data_seq : str
            Path to txt file containing list of which data sequences to use
        """
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        with open(data_seq, 'r') as file:
            paths = file.readlines()

        # initialise counter for sample indices
        sample_idx = 0

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
                img0_path = os.path.join(image_dir, image_files[i])
                img1_path = os.path.join(image_dir, image_files[i + 1])
                sample = {
                    'img0_path': img0_path,
                    'img1_path': img1_path,
                    'pose_file': pose_file,
                    'pose_indices': (i, i + 1)
                }

                # load images
                img_0 = load_image(img0_path)
                img_1 = load_image(img1_path)

                # load poses
                poses = np.loadtxt(pose_file)
                pose_0, pose_1 = poses[i], poses[i + 1]

                # get coordinates of matched keypoints
                kpts_coords = self.feature_matcher.match_img_pair(img_0, img_1)

                # convert poses to transformation matrices
                transformation_mat_0 = self.pose_transforms.pose_to_mat(pose_0[:3], pose_0[3:])
                transformation_mat_1 = self.pose_transforms.pose_to_mat(pose_1[:3], pose_1[3:])

                # get relative pose
                rel_translation, rel_rot_six_d = self.pose_transforms.get_relative_pose(transformation_mat_0, transformation_mat_1)

                # convert to tensors
                kpts_coords = kpts_coords.float()
                rel_translation = torch.from_numpy(rel_translation).float()
                rel_rot_six_d = torch.tensor(rel_rot_six_d).float()

                # save preprocessed data
                output_path = os.path.join(self.preprocessed_dir, f'sample_{sample_idx}.pt')
                torch.save({
                    'kpts_coords': kpts_coords,
                    'rel_translation': rel_translation,
                    'rel_rot_six_d': rel_rot_six_d
                }, output_path)
                
                # collect saved file paths
                self.samples.append(output_path)

                # increment counter
                sample_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load the preprocessed data

        Parameters
        ----------
        idx : int
            Index of the preprocessed sample

        Returns
        -------
        kpts_coords, rel_translation, rel_rot_six_d : torch.Tensor, torch.Tensor, torch.Tensor
            Preprocessed keypoint coordinates, relative translation, and relative rotation
        """
        sample = torch.load(self.samples[idx])

        return sample['kpts_coords'], sample['rel_translation'], sample['rel_rot_six_d']
"""
Evaluation script for pose estimation model
Performs inference using pose estimation model to get estimated relative poses
Uses first pose from ground truth txt file for initial pose
Updates current pose using each relative pose
Returns txt file in TUM format to compare estimated poses with gt

Author: Saifullah Ijaz
Date: 09/09/2024
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
import numpy as np
from pose_estimation import PoseEstimationTransformer
from feature_matching import FeatureMatching
from pose_transforms import PoseTransforms

# add root level directory to path to access Datasets folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LightGlue.lightglue.utils import load_image

class VisualOdometry():
    def __init__(self, image_dir, gt_poses, model_path, output_file, device) -> None:
        self.image_dir = image_dir
        self.output_file = output_file
        self.feature_matcher = FeatureMatching(device)
        self.transforms = PoseTransforms()

        # initialise first pose to the first one used in the ground truth txt file
        poses = np.loadtxt(gt_poses)
        initial_pose = poses[0]
        self.initial_translation = [initial_pose[0], initial_pose[1], initial_pose[2]]
        self.initial_rotation = [initial_pose[3], initial_pose[4], initial_pose[5], initial_pose[6]]
        self.cur_pose = self.transforms.pose_to_mat(self.initial_translation, self.initial_rotation)
        
        # initialise pose estimation model
        self.model = PoseEstimationTransformer().to(device)
        weights = torch.load(model_path)
        self.model.load_state_dict(weights['model_state_dict'])
        self.model.eval()
    
    def odometry(self):
        image_files = sorted([img for img in os.listdir(self.image_dir) if img.endswith('.png')])

        with open(self.output_file, 'w') as f:
            tx, ty, tz = self.initial_translation[0], self.initial_translation[1], self.initial_translation[2]
            qx, qy, qz, qw = self.initial_rotation[0], self.initial_rotation[1], self.initial_rotation[2], self.initial_rotation[3]
            f.write(f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

            for i in range(len(image_files) - 1):
                img0_path = os.path.join(self.image_dir, image_files[i])
                img1_path = os.path.join(self.image_dir, image_files[i + 1])

                # load images
                img_0 = load_image(img0_path)
                img_1 = load_image(img1_path)

                # get coordinates of matched keypoints
                kpts_coords = self.feature_matcher.match_img_pair(img_0, img_1).float().unsqueeze(0)

                # create a mask (1 for valid, 0 for invalid)
                mask = torch.ones(kpts_coords.shape[1], dtype=torch.bool, device=kpts_coords.device).unsqueeze(0)

                # forward pass
                pred_translation, pred_rotation = self.model(kpts_coords, mask)
                pred_translation, pred_rotation = pred_translation.squeeze(0), pred_rotation.squeeze(0)

                # move to CPU
                translation = pred_translation.detach().cpu().numpy()
                rotation_six_d = list(pred_rotation.detach().cpu().numpy())

                # convert rotations
                rotation_mat = self.transforms.six_d_to_rot_matrix(rotation_six_d)
                rotation_quat = self.transforms.rot_mat_to_quat(rotation_mat)

                # get relative pose
                rel_transformation_mat = self.transforms.pose_to_mat(translation, rotation_quat)

                # update current cumulative pose
                self.cur_pose = np.dot(self.cur_pose, rel_transformation_mat)

                # extract translation from the current pose matrix
                tx, ty, tz = self.cur_pose[0, 3], self.cur_pose[1, 3], self.cur_pose[2, 3]

                # extract rotation matrix and convert to quaternion
                rotation_matrix = self.cur_pose[:3, :3]
                qx, qy, qz, qw = self.transforms.rot_mat_to_quat(rotation_matrix)

                # write to file in the form: idx tx ty tz qx qy qz qw
                f.write(f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device} \n')

    image_dir = '../Datasets/TartanAir/MH005/'
    gt_poses = '../Datasets/TartanAir/mono_gt/MH005.txt'
    model_path = 'forest_lg_pose_est.pth'
    output_file = '../Datasets/TartanAir/mono_gt/MH005_est.txt'

    vo = VisualOdometry(image_dir, gt_poses, model_path, output_file, device)

    vo.odometry()

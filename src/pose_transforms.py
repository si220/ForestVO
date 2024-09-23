"""
Pose Transformations

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseTransforms():
    def __init__(self) -> None:
        pass

    # convert [x,y,z] translation and [x,y,z,w] rotation into 4x4 transformation matrix
    def pose_to_mat(self, translation, quat):
        transformation_mat = np.eye(4)
        rotation_mat = R.from_quat(quat).as_matrix()
        transformation_mat[:3, 3] = translation
        transformation_mat[:3, :3] = rotation_mat

        return transformation_mat
    
    # convert 3x3 rotation matrix into quaternion
    def rot_mat_to_quat(self, rot_mat):
        return R.from_matrix(rot_mat).as_quat()

    # compute relative transformation between 2 transformation matrices
    def get_relative_pose(self, T0, T1):
        # get relative pose
        T_rel = np.linalg.inv(T0) @ T1

        # get relative translation vector and relative 3x3 rotation matrix
        translation = T_rel[:3, 3]
        rotation = T_rel[:3, :3]

        # convert relative rotation to 6D representation (first 2 cols of 3x3 rotation matrix)
        first_col = rotation[:, :1].flatten()
        second_col = rotation[:, 1:2].flatten()
        rotation_six_d = [first_col[0], first_col[1], first_col[2], second_col[0], second_col[1], second_col[2]]

        return translation, rotation_six_d
    
    # convert 6D rotation back to 3x3 rotation matrix using Gram-Schmidt process
    def six_d_to_rot_matrix(self, six_d):
        # reshape the 6D vector into two 3D column vectors
        r1 = six_d[:3]
        r2 = six_d[3:]
        
        # normalise the first vector
        r1 = r1 / np.linalg.norm(r1)
        
        # make the second vector orthogonal to the first
        r2 = r2 - np.dot(r2, r1) * r1
        r2 = r2 / np.linalg.norm(r2)
        
        # third vector is the cross product of the first two
        r3 = np.cross(r1, r2)
        
        # combine to form the rotation matrix
        rotation_matrix = np.column_stack([r1, r2, r3])

        return rotation_matrix
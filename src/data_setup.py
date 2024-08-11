"""
script to create training, validation and test sets for the pose estimation model

passes image pairs through SuperPoint and LightGlue to get coordinates of matched features in both frames
gets corresponding ground truth relative poses for each pair of frames

each data sample is provided as an npz file containing the following:

points0 -> np array containing coordinates of matched kpts in img0
points1 -> np array containing coordinates of matched kpts in img1
rel_translation -> relative translation given as [x,y,z]
rel_rot_six_d -> relative rotation given as 6D representation

inputs:
    train_seq (path to txt file listing directories to use as training samples)
    val_seq (path to txt file listing directories to use as validation samples)
    test_seq (path to txt file listing directories to use as testing samples)

    train_dir (path to directory to save training data)
    val_dir (path to directory to save validation data)
    test_dir (path to directory to save testing data)

outputs:
    train_dir (directory containing training data)
    val_dir (directory containing validation data)
    test_dir (directory containing testing data)

Author: Saifullah Ijaz
Date: 25/07/2024
"""

# import required libraries and functions
from globals import *
from feature_matching import *

class create_dataset:
    def __init__(self, train_seq, val_seq, test_seq, train_dir, val_dir, test_dir):
        self.train_seq = train_seq
        self.val_seq = val_seq
        self.test_seq = test_seq

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    # convert [x,y,z] translation and [x,y,z,w] rotation into 4x4 transformation matrix
    def pose_to_mat(self, translation, quat):
        transformation_mat = np.eye(4)
        rotation_mat = R.from_quat(quat).as_matrix()
        transformation_mat[:3, 3] = translation
        transformation_mat[:3, :3] = rotation_mat

        return transformation_mat

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
    
    def create_data(self, data_seq, output_dir):
        with open(data_seq, 'r') as file:
            paths = file.readlines()

        # clear existing dataset directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # make new dataset directory
        os.mkdir(output_dir)

        # loop through directories specified for sample sequences
        for idx, path in enumerate(paths):
            path = path.strip()

            # use the correct image folder
            if 'left' in path:
                image_dir = os.path.join(path, 'image_left/')
                pose_file = os.path.join(path, 'pose_left.txt')

            else:
                image_dir = os.path.join(path, 'image_right/')
                pose_file = os.path.join(path, 'pose_right.txt')

            poses = np.loadtxt(pose_file)
            image_files = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])

            for i in range(len(image_files) - 1):
                img0_path = os.path.join(image_dir, image_files[i])
                img1_path = os.path.join(image_dir, image_files[i + 1])

                img_0 = load_image(img0_path)
                img_1 = load_image(img1_path)

                # get coordinates of matched kpts in both images
                _, _, _, _, _, _, points0, points1 = match_img_pair(img_0, img_1)

                # convert to np arrays
                points0, points1 = points0.detach().cpu().numpy(), points1.detach().cpu().numpy()

                # get poses from ground truth pose txt file
                pose_0, pose_1 = poses[i], poses[i+1]
                transformation_mat_0 = self.pose_to_mat(pose_0[:3], pose_0[3:])
                transformation_mat_1 = self.pose_to_mat(pose_1[:3], pose_1[3:])

                # get relative pose
                rel_translation, rel_rot_six_d = self.get_relative_pose(transformation_mat_0, transformation_mat_1)

                # save the data as an npz file in the output directory
                npz_filename = f"sample_{idx}_{i}.npz"
                npz_path = os.path.join(output_dir, npz_filename)
                np.savez(npz_path,
                        points0=points0,
                        points1=points1,
                        rel_translation=rel_translation,
                        rel_rot_six_d=rel_rot_six_d)

if __name__ == "__main__":
    # txt files containing paths to train, val and test sequences
    train_seq = 'dataset/train.txt'
    val_seq = 'dataset/val.txt'
    test_seq = 'dataset/test.txt'

    # output folders to store data
    train_dir = '../Datasets/train/'
    val_dir = '../Datasets/val/'
    test_dir = '../Datasets/test/'

    dataset = create_dataset(train_seq, val_seq, test_seq, train_dir, val_dir, test_dir)
    
    # create training data
    dataset.create_data(train_seq, train_dir)

    # create validation data
    dataset.create_data(val_seq, val_dir)

    # create testing data
    dataset.create_data(test_seq, test_dir)

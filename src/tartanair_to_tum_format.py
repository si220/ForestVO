"""
Convert ground truth TartanAir pose txt files to TUM format to use with evo evaluation tool
Adds timestamp column at the start to match format expected by evo

inputs:
    data_path (path to TartanAir dataset)

outputs:
    pose_left_tum.txt and pose_right_tum.txt (pose txt files in TUM format)

usage:
    python3 tartanair_to_tum_format.py --data_path <absolute path to TartanAir dataset directory>

Author: Saifullah Ijaz
Date: 16/07/2024
"""


from globals import *

def convert_to_tum_format(input_file, output_file):
    # parse input
    poses = np.loadtxt(input_file)

    with open(output_file, 'w') as f:
        for i, pose in enumerate(poses):
            # translation given as first 3 cols of poses
            tx, ty, tz = pose[:3]

            # rotation given as last 4 cols of poses
            qx, qy, qz, qw = pose[3:]

            # create fake timestamp
            timestamp = i * 0.1

            # write the pose in TUM format with timestamp as first col
            f.write(f'{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n')

# function to recurse through directory and convert all pose txt files to TUM format
def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        print(f'checking directory: {root}')

        for file in files:
            if file in ['pose_left.txt', 'pose_right.txt']:
                input_file = os.path.join(root, file)
                output_file = os.path.join(root, f'{os.path.splitext(file)[0]}_tum.txt')

                print(f'converting {input_file} to {output_file}')
                convert_to_tum_format(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert TartanAir pose files to TUM format')
    parser.add_argument('--data_path', type=str, help='path to TartanAir dataset')
    
    args = parser.parse_args()
    
    process_directory(args.data_path)
    
    print(f'conversion complete')

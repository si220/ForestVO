"""
Perform LightGlue feature matching on directory of image files

inputs:
    path_to_images (path to directory of images)

outputs:
    draw_matches() -> OpenCV visualisation of matched features
    match_img_pair() -> images, matched keypoints and coordinates
    visualise_feature_matching() -> loops through image files and visualises matched features

usage:
    python3 feature_matching.py

Author: Saifullah Ijaz
Date: 23/07/2024
"""

from globals import *

# disable gradient calculation for faster inference
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

# initialise SuperPoint and LightGlue
feature_detector = SuperPoint(max_num_keypoints=2048).eval().to(device)
feature_matcher = LightGlue(features="superpoint").eval().to(device)

# path to the directory of images
path_to_images = '../Datasets/TartanAir/gascola_Easy_image_left/gascola/Easy/P001/image_left/'
input_images = Path(path_to_images)

# sort images
image_files = sorted(input_images.glob("*.png"))

# function to visualise feature matches using OpenCV
def draw_matches(img0, img1, kpts0, kpts1, matches, colour=None):
    # create a new output image that concatenates the two images together
    rows0, cols0 = img0.shape[:2]
    rows1, cols1 = img1.shape[:2]
    out = np.zeros((max([rows0, rows1]), cols0 + cols1, 3), dtype='uint8')
    out[:rows0, :cols0, :] = img0
    out[:rows1, cols0:cols0+cols1, :] = img1

    # use green lines to represent feature matches
    if colour is None:
        colour = (0, 255, 0)

    for i in range(len(matches)):
        # get the matched keypoints for both images
        img0_idx = matches[i, 0]
        img1_idx = matches[i, 1]

        # x = columns, y = rows
        (x0, y0) = kpts0[img0_idx].cpu().numpy()
        (x1, y1) = kpts1[img1_idx].cpu().numpy()

        # draw a small circle at both co-ordinates
        cv2.circle(out, (int(x0), int(y0)), 4, colour, 1)
        cv2.circle(out, (int(x1) + cols0, int(y1)), 4, colour, 1)

        # draw a line between the two points
        cv2.line(out, (int(x0), int(y0)), (int(x1) + cols0, int(y1)), colour, 1)

    return out

# match features between pair of images
def match_img_pair(image0, image1):
    # convert images to OpenCV format
    img0 = (image0.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img1 = (image1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # SuperPoint feature detection
    feats0 = feature_detector.extract(image0.to(device))
    feats1 = feature_detector.extract(image1.to(device))

    # LightGlue feature matching
    matches01 = feature_matcher({"image0": feats0, "image1": feats1})

    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # get keypoints and matches
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    # get coordinates of matched keypoints in both images
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in first image, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in second image, shape (K,2)

    return img0, img1, kpts0, kpts1, matches01, matches, points0, points1

def visualise_feature_matching(image_files):
    # loop through the image pairs and perform feature matching
    for i in range(len(image_files) - 1):
        image0_path = image_files[i]
        image1_path = image_files[i + 1]

        # load images
        image0 = load_image(image0_path)
        image1 = load_image(image1_path)

        # match features across image pair
        img0, img1, kpts0, kpts1, matches01, matches, points0, points1 = match_img_pair(image0, image1)

        # draw matches
        img_matches = draw_matches(img0, img1, kpts0, kpts1, matches)

        # add text
        cv2.putText(img_matches, f'Stop after {matches01["stop"]} layers', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # display the image
        cv2.imshow('Matches', img_matches)

        # exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    visualise_feature_matching(image_files)

    # close all OpenCV windows
    cv2.destroyAllWindows()

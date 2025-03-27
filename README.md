# ForestVO
This repo contains the implementation of the pose estimation model for the "**ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue**" paper which has been accepted for publication in the IEEE Robotics and Automation Letters journal. The main repo for the project which contains the ForestGlue implementation can be found [here](https://github.com/AerialRoboticsGroup/forest-vo)

## Publication
If you use this code in an academic context, please cite the following RAL 2025 paper.

T. Pritchard (equal contribution), S. Ijaz (equal contribution), R. Clark, and BB. Kocer, "**ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue**," Robotics and Automation Letters (RA-L). 2025.

```
@article{pritchard2025forestvo,
  title={ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue},
  author={Pritchard, Thomas and Ijaz, Saifullah and Clark, Ronald and Kocer, Basaran Bahadir},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

## Installation
Clone the repo recursively:
```
git clone --recursive https://github.com/si220/ForestVO.git
```

Follow the setup instructions from https://github.com/cvg/LightGlue to get LightGlue working.

Install the required packages for this repo by running:
```
pip install -r requirements.txt
```
## Training

### Setup Dataset
Follow the TartanAir dataset format, where each sequence contains images in an image_left/ or image_right/ folder with a corresponding pose_left.txt or pose_right.txt file containing the ground truth absolute poses. See https://github.com/castacks/tartanair_tools for more info.

Modify the [train.txt](ForestVO/src/dataset/train.txt), [val.txt](ForestVO/src/dataset/val.txt) and [test.txt](ForestVO/src/dataset/test.txt) files with the sequences you want to use for the trainining, validation and test sets respectively.

In [train.py](ForestVO/src/train.py) set the desired location to save the pre-processed training and validation data:
```python
preprocessed_train_dir = "../Datasets/forest_lg_train/"
preprocessed_val_dir = "../Datasets/forest_lg_val/"
```

Each sample in the generated training and validation set consists of the 2D coordinates of the matched keypoints between an image pair as well as the corresponding ground truth relative camera pose between frames. The ground truth pose is split into the 3D relative translation and 6D relative rotation. 

### Train the Pose Estimation Model
Modify the hyperparameters in [train.py](ForestVO/src/train.py) based on your requirements.

Run:
```
python3 train.py
```

To continue training from a saved checkpoint, set the checkpoint path in train.py
```python
checkpoint_path = "checkpoints/epoch_50.pth"
```

To visualise the losses in your browser, launch tensorboard by running the following:
```
tensorboard --logdir=experiments/
```

## Evaluation
We provide the weights for pre-trained pose estimation models using both the default LightGlue model and the ForestGlue model which is a LightGlue model that has been finetuned on forest data.  

## Pose Estimation Model using Default LightGlue
To use the pre-trained pose estimation model which uses the default LightGlue model, set the number of attention heads and transformer encoder layers to 2 in the [pose_estimation.py](ForestVO/src/pose_estimation.py) file:

```python
def __init__(self, input_dim=4, d_model=128, nhead=2, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
```

Modify the following, making sure to specify the default model in [eval.py](ForestVO/src/eval.py):
```python
image_dir = '../Datasets/TartanAir/MH005/'
gt_poses = '../Datasets/TartanAir/mono_gt/MH005.txt'
model_path = 'default_lg_pose_est.pth'
output_file = '../Datasets/TartanAir/mono_gt/MH005_est.txt'
```

To generate the estimated pose txt file, run:
```
python3 eval.py
```

To evaluate against the ground truth pose txt files, see https://github.com/castacks/tartanair_tools for more info.

## Pose Estimation Model using ForestGlue
To use the pre-trained pose estimation model which uses the default LightGlue model, set the number of attention heads and transformer encoder layers to 4 in the [pose_estimation.py](ForestVO/src/pose_estimation.py) file:

```python
def __init__(self, input_dim=4, d_model=128, nhead=4, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
```

Modify the following, making sure to specify the forest model in [eval.py](ForestVO/src/eval.py):
```python
image_dir = '../Datasets/TartanAir/MH005/'
gt_poses = '../Datasets/TartanAir/mono_gt/MH005.txt'
model_path = 'forest_lg_pose_est.pth'
output_file = '../Datasets/TartanAir/mono_gt/MH005_est.txt'
```

Modify lines 408-427 in [lightglue.py](ForestVO/LightGlue/lightglue/lightglue.py) making sure to specify the path to [lightglue_checkpoint_best.tar](ForestVO/src/lightglue_checkpoint_best.tar), the tar file containing the weights for the ForestGlue model.

Original:

```python
state_dict = None
if features is not None:
    fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
    state_dict = torch.hub.load_state_dict_from_url(
        self.url.format(self.version, features), file_name=fname
    )
    self.load_state_dict(state_dict, strict=False)
elif conf.weights is not None:
    path = Path(__file__).parent
    path = path / "weights/{}.pth".format(self.conf.weights)
    state_dict = torch.load(str(path), map_location="cpu")

if state_dict:
    # rename old state dict entries
    for i in range(self.conf.n_layers):
        pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    self.load_state_dict(state_dict, strict=False)
```

Modified:

```python
# state_dict = None
# if features is not None:
#     fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
#     state_dict = torch.hub.load_state_dict_from_url(
#         self.url.format(self.version, features), file_name=fname
#     )
#     self.load_state_dict(state_dict, strict=False)
# elif conf.weights is not None:
#     path = Path(__file__).parent
#     path = path / "weights/{}.pth".format(self.conf.weights)
#     state_dict = torch.load(str(path), map_location="cpu")

checkpoint_path = f"path to lightglue_checkpoint_best.tar"
state_dict = torch.load(checkpoint_path, map_location="cpu")['model']

if state_dict:
    # rename old state dict entries
    pattern = f"matcher.", f""
    state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    pattern = f"extractor.", f""
    state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    # for i in range(self.conf.n_layers):
    #     pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
    #     state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    #     pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
    #     state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    self.load_state_dict(state_dict, strict=False)
```

To generate the estimated pose txt file, run:
```
python3 eval.py
```

"""
Training script for pose estimation model

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pose_estimation import PoseEstimationTransformer
from feature_matching import FeatureMatching
from pose_transforms import PoseTransforms
from dataset import PoseEstimationDataset

def collate(batch):
    kpts_coords = [item[0].float() for item in batch]
    rel_translations = torch.stack([item[1].float() for item in batch])
    rel_rot_six_d = torch.stack([item[2].float() for item in batch])
    
    return kpts_coords, rel_translations, rel_rot_six_d

def train_model(model, dataloader, device, num_epochs=10, learning_rate=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    translation_criterion = nn.MSELoss()
    rotation_criterion = nn.MSELoss()

    # initialise tensorboard writer
    writer = SummaryWriter(log_dir="experiments/relative_pose_regression")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (kpts_coords_batch, rel_translation, rel_rot_six_d) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # move data to device
            kpts_coords_batch = [kpts.to(device) for kpts in kpts_coords_batch]
            rel_translation = rel_translation.to(device)
            rel_rot_six_d = rel_rot_six_d.to(device)
            
            # forward pass
            pred_translation, pred_rotation = model(kpts_coords_batch)
            
            # compute loss
            translation_loss = translation_criterion(pred_translation, rel_translation)
            rotation_loss = rotation_criterion(pred_rotation, rel_rot_six_d)
            loss = translation_loss + rotation_loss
            
            # backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # store losses for tensorboard writer for every batch
            writer.add_scalar("Loss/Train_Batch", loss.item(), epoch * len(dataloader) + batch_idx)
        
        # store avg loss for each epoch
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/Train_Epoch", avg_loss, epoch)

    # write all pending events to disk and close writer
    writer.flush()
    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    # instantiate the FeatureMatching and PoseTransforms classes
    feature_matcher = FeatureMatching(device)
    pose_transforms = PoseTransforms()

    # initialise dataset
    data_seq = "dataset/train.txt"
    dataset = PoseEstimationDataset(data_seq, feature_matcher, pose_transforms, device)
    
    # initialise dataloader
    batch_size = 128
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    input_dim = 4  # x1, y1, x2, y2
    d_model = 256
    model = PoseEstimationTransformer(input_dim, d_model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_trainable_params = {num_params}')

    train_model(model, dataloader, device, num_epochs=3, learning_rate=0.001)
    
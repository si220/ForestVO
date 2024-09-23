"""
Training script for pose estimation model

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
from pose_estimation import PoseEstimationTransformer
from feature_matching import FeatureMatching
from pose_transforms import PoseTransforms
from dataset import PoseEstimationDataset

# add root level directory to path to access Datasets folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def collate(batch):
    kpts_coords_batch = [item[0].float() for item in batch]
    rel_translation = torch.stack([item[1].float() for item in batch])
    rel_rot_six_d = torch.stack([item[2].float() for item in batch])

    # pad kpts_coords_batch to have the same length
    kpts_coords_batch_padded = pad_sequence(kpts_coords_batch, batch_first=True, padding_value=0)

    # create lengths tensor for padded sequence
    lengths = torch.tensor([kpts.size(0) for kpts in kpts_coords_batch], dtype=torch.long)

    # create masks to ignore padded values
    masks = torch.arange(kpts_coords_batch_padded.size(1)).expand(len(lengths), -1) < lengths.unsqueeze(1)

    # 1 means actual data, 0 means padded value to ignore
    masks = masks.bool()

    return kpts_coords_batch_padded, lengths, masks, rel_translation, rel_rot_six_d

def save_checkpoint(epoch, model, optimiser, loss, file_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, file_path)

def load_checkpoint(model, optimiser, file_path, device):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        # move optimiser state to the correct device
        for state in optimiser.state.values():
            if isinstance(state, dict):
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {file_path}, Epoch: {epoch}, Loss: {loss}")

        return epoch, loss
    
    else:
        print(f"No checkpoint found at {file_path}")

        return None, None

def train_model(model, train_loader, val_loader, device, optimiser, num_epochs=10, checkpoint_dir="checkpoints", start_epoch=0, beta=100):
    model = model.to(device)
    translation_criterion = nn.MSELoss()
    rotation_criterion = nn.MSELoss()

    # create directory for checkpoints if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # initialise tensorboard writer (append mode to avoid overwriting existing data)
    writer = SummaryWriter(log_dir="experiments/forest_lg", purge_step=start_epoch * len(train_loader))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm progress bar
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (kpts_coords_batch_padded, lengths, masks, rel_translation, rel_rot_six_d) in train_loader_tqdm:
            optimiser.zero_grad()

            # move data to the correct device (GPU or CPU)
            kpts_coords_batch_padded = kpts_coords_batch_padded.to(device)
            rel_translation = rel_translation.to(device)
            rel_rot_six_d = rel_rot_six_d.to(device)
            masks = masks.to(device)

            # forward pass
            pred_translation, pred_rotation = model(kpts_coords_batch_padded, masks)

            # compute loss
            translation_loss = translation_criterion(pred_translation, rel_translation)
            rotation_loss = rotation_criterion(pred_rotation, rel_rot_six_d)
            loss = translation_loss + (beta * rotation_loss)

            # backprop
            loss.backward()

            # ensure optimiser states are on the correct device
            for param_group in optimiser.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad = param.grad.to(device)

            optimiser.step()

            total_train_loss += loss.item()

            # compare predicted and gt poses
            if batch_idx == 0:
                print(f'pred_translation_train = {pred_translation[0]}')
                print(f'gt_translation_train = {rel_translation[0]}')
                print(f'pred_rotation_train = {pred_rotation[0]}')
                print(f'gt_rotation_train = {rel_rot_six_d[0]} \n')

            # update tqdm progress bar with loss
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

            # store losses for tensorboard writer for every batch, with global step
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)

        # store avg training loss for each epoch
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        # validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch+1}/{num_epochs}")

            for batch_idx, (kpts_coords_batch_padded, lengths, masks, rel_translation, rel_rot_six_d) in val_loader_tqdm:
                # move data to the correct device (GPU or CPU)
                kpts_coords_batch_padded = kpts_coords_batch_padded.to(device)
                rel_translation = rel_translation.to(device)
                rel_rot_six_d = rel_rot_six_d.to(device)
                masks = masks.to(device)

                # forward pass
                pred_translation, pred_rotation = model(kpts_coords_batch_padded, masks)

                # compute loss
                translation_loss = translation_criterion(pred_translation, rel_translation)
                rotation_loss = rotation_criterion(pred_rotation, rel_rot_six_d)
                loss = translation_loss + (beta * rotation_loss)

                total_val_loss += loss.item()

                # compare predicted and gt poses
                if batch_idx == 0:
                    print(f'pred_translation_val = {pred_translation[0]}')
                    print(f'gt_translation_val = {rel_translation[0]}')
                    print(f'pred_rotation_val = {pred_rotation[0]}')
                    print(f'gt_rotation_val = {rel_rot_six_d[0]} \n')

                # update tqdm progress bar with validation loss
                val_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        # store avg validation loss for each epoch
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # save the model checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        save_checkpoint(epoch+1, model, optimiser, avg_val_loss, checkpoint_path)

    # write all pending events to disk and close writer
    writer.flush()
    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    # check if the preprocessed directories exist, preprocess only if they don't
    preprocessed_train_dir = "../Datasets/forest_lg_train/"
    preprocessed_val_dir = "../Datasets/forest_lg_val/"

    if not os.path.exists(preprocessed_train_dir):
        print("preprocessing training dataset")

        train_dataset = PoseEstimationDataset(
            data_seq="dataset/train.txt",
            feature_matcher=FeatureMatching(device),
            pose_transforms=PoseTransforms(),
            device=device,
            preprocessed_dir=preprocessed_train_dir,
            preprocess=True
        )

    else:
        print("loading preprocessed training dataset")

    if not os.path.exists(preprocessed_val_dir):
        print("preprocessing validation dataset")

        val_dataset = PoseEstimationDataset(
            data_seq="dataset/val.txt",
            feature_matcher=FeatureMatching(device),
            pose_transforms=PoseTransforms(),
            device=device,
            preprocessed_dir=preprocessed_val_dir,
            preprocess=True
        )

    else:
        print("loading preprocessed validation dataset")

    # load preprocessed datasets for training
    train_dataset = PoseEstimationDataset(preprocessed_dir=preprocessed_train_dir, preprocess=False)
    val_dataset = PoseEstimationDataset(preprocessed_dir=preprocessed_val_dir, preprocess=False)

    # initialise dataloaders
    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # x0, y0, x1, y1
    input_dim = 4
    d_model = 128
    model = PoseEstimationTransformer(input_dim, d_model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_trainable_params = {num_params}')

    # initialise optimiser and load the model
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    checkpoint_path = "checkpoints/epoch_49.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimiser, checkpoint_path, device)
        start_epoch = start_epoch if start_epoch is not None else 0

    # train model
    train_model(model, train_loader, val_loader, device, optimiser, num_epochs=100, start_epoch=start_epoch, beta=100)

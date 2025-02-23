import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import cv2
import numpy as np

# Hyperparameters
size = 128  # Crop size
full_size = 340  # Original image size
batch_size = 1
learning_rate = 0.0001
num_epochs = 40

# ----------------------------------------------------------
# **Vision Dataset using PyTorch's DataLoader**
class VisionDataset(Dataset):
    def __init__(self, sessions_dir, validation_split=0.1, train=True):
        self.sessions_dir = sessions_dir
        self.images = []
        self.labels = []
        self.crops = []

        for filename in os.listdir(sessions_dir):
            if filename.startswith("f2_") and filename.endswith(".png"):
                parts = filename.replace(".png", "").split("_")
                if len(parts) == 7:
                    _, _, eye_x, eye_y, head_x, head_y, head_z = parts
                    json_path = os.path.join(sessions_dir, filename.replace(".png", ".json"))

                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            json_data = json.load(f)
                            crop_pos = json_data["labels"][0]["eye-center"]
                    else:
                        print(f"Warning: No annotation for {filename}")
                        crop_pos = [0.0, 0.0]

                    self.images.append(os.path.join(sessions_dir, filename))
                    self.labels.append([float(eye_x), float(eye_y)])
                    self.crops.append(crop_pos)

        # Shuffle dataset
        combined = list(zip(self.images, self.labels, self.crops))
        random.shuffle(combined)
        self.images, self.labels, self.crops = zip(*combined)

        # Split into train and validation
        split_idx = int(len(self.images) * (1 - validation_split))
        if train:
            self.images, self.labels, self.crops = self.images[:split_idx], self.labels[:split_idx], self.crops[:split_idx]
        else:
            self.images, self.labels, self.crops = self.images[split_idx:], self.labels[split_idx:], self.crops[split_idx:]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0, 255] to [0, 1]
        ])

    def center_crop(self, image, crop_offset):
        """Dynamically crops based on user-provided targeting (-0.5 to 0.5 range)."""
        crop_x = int((crop_offset[0] + 0.5) * full_size)
        crop_y = int((crop_offset[1] + 0.5) * full_size)

        start_x = max(0, min(crop_x - size // 2, full_size - size))
        start_y = max(0, min(crop_y - size // 2, full_size - size))

        return image[start_y:start_y + size, start_x:start_x + size]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        # this seems to have no effect!  so it is not learning from these parameters of crop offset
        crop_offset = torch.tensor(self.crops[idx], dtype=torch.float32)
        #crop_offset = torch.tensor([0.0, 0.0], dtype=torch.float32)
        
        # Load and process image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image {img_path}")

        image = self.center_crop(image, self.crops[idx])
        image = self.transform(image)  # Convert to tensor [1, 128, 128]

        return image, crop_offset, label

# ----------------------------------------------------------
# **Define PyTorch Model**
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * (size // 4) * (size // 4), 512)  # Adjust for 2 max-pool layers
        self.fc2 = nn.Linear(512 + 2, 512)  # Merge crop offset
        self.fc3 = nn.Linear(512, 2)  # Output gaze X, Y

    def forward(self, image, crop_offset):
        x = self.pool(self.relu(self.conv1(image)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))

        # Merge crop offset
        x = torch.cat((x, crop_offset), dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_errors = torch.zeros(2, device=device)  # Accumulate total X, Y errors

        print(f'--- Epoch [{epoch+1}/{num_epochs}] ---')

        for batch_idx, (images, crop_offsets, labels) in enumerate(train_loader):
            images, crop_offsets, labels = images.to(device), crop_offsets.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, crop_offsets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # **Accumulate batch errors**
            batch_error = torch.mean(torch.abs(outputs - labels), dim=0)  # Mean error for this batch
            total_errors += batch_error  # Add to total

        # **Compute Averages**
        avg_loss = total_loss / len(train_loader)
        avg_errors = (total_errors / len(train_loader)).detach().cpu().numpy()  # Final epoch-wide X, Y error

        print(f'Train Loss: {avg_loss:.4f} | '
            f'Error: X={avg_errors[0]:.4f}, Y={avg_errors[1]:.4f}')
        print('------------------------------------------')

        # Validation Step
        model.eval()
        val_loss = 0
        total_val_errors = torch.zeros(2, device=device)
        with torch.no_grad():
            for val_images, val_crop_offsets, val_labels in val_loader:
                val_images, val_crop_offsets, val_labels = val_images.to(device), val_crop_offsets.to(device), val_labels.to(device)
                val_outputs = model(val_images, val_crop_offsets)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                total_val_errors += torch.mean(torch.abs(val_outputs - val_labels), dim=0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_errors = total_val_errors.cpu().numpy() / len(val_loader)

        print(f'Validation Loss: {avg_val_loss:.4f} | '
              f'Validation Error: X={avg_val_errors[0]:.4f}, Y={avg_val_errors[1]:.4f}')
        print('------------------------------------------\n')


# ----------------------------------------------------------
# **Run Training**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id = 'wkjykq'
train_dataset = VisionDataset(f'sessions/{id}', train=True)
val_dataset = VisionDataset(f'sessions/{id}', train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = VisionModel()
train(model, train_loader, val_loader, device)

# Save model
torch.save(model.state_dict(), "vision_model.pth")

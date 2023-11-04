import torch
import torch.nn as nn
from   torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from   torch.utils.data import DataLoader, Dataset
from   torchvision import transforms
from   PIL import Image
import numpy as np
import json
import os

# Define the path to the directory containing the JSON files
W = 256
H = 256
lr = 0.0001

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPUs available on this system.")

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen_path = './gen'
fields  = ['qx', 'qy', 'qz', 'qw']
labels  = []
sources = []

p = os.getcwd()
if os.path.exists('build'):
    os.chdir('build')

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((H, W)),  # Assuming H and W are defined
    transforms.ToTensor(),
])

# iterate over the files in the directory
for f in os.listdir(gen_path):
    if not f.endswith('.json'): continue
    count = 0
    with open(f'{gen_path}/{f}', 'r', encoding='utf-8') as fdata:
        try:
            js = json.load(fdata)
        except Exception as e:
            print(f"error occurred with file {f}: {e} (skipping...)")
            continue

        lv = []
        for field in fields:
            v = js['labels'][field]
            assert(v >= -1024 and v <= 1024)
            lv.append(v)
        
        source = f'{gen_path}/{js["source"]}'
        if not os.path.exists(source):
            continue
        assert(os.path.exists(source))
        labels.append(lv)
        sources.append(source)

# Convert labels to a numpy array
labels = np.array(labels)

print(labels.shape) # (2200, 8)
print(len(sources)) #  2200


# Instantiate your dataset and dataloader
dataset = CustomDataset(sources, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# cnn model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, len(fields))  # No activation, directly outputting the regression values
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# transformers ViT
class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, nheads, num_layers, num_outputs):
        super(ViT, self).__init__()
        self.patch_size  = patch_size
        self.num_patches = (W // patch_size) * (H // patch_size)
        self.emb_size    = emb_size
        self.patch_emb   = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=emb_size, 
                                   kernel_size=patch_size, 
                                   stride=patch_size)
        self.pos_emb     = nn.Parameter(torch.randn(1, self.num_patches+1, emb_size))
        self.cls_token   = nn.Parameter(torch.randn(1, 1, emb_size))
        encoder_layers   = TransformerEncoderLayer(d_model=emb_size, nhead=nheads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.mlp_head    = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_outputs)  # Adjusted for arbitrary number of outputs
        )
    
    def forward(self, x):
        x = self.patch_emb(x)       # Create patch embeddings
        x = x.flatten(2)            # Flatten the patch dimensions
        x = x.transpose(1, 2)       # NxCxS to NxSxC
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # Duplicate cls token for batch
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate cls token with patch embeddings
        x = x + self.pos_emb        # Add positional embeddings
        x = self.transformer_encoder(x)
        cls_token_final = x[:, 0]   # We only use the cls token to classify
        return self.mlp_head(cls_token_final)

vit = ViT(
    in_channels=3,
    patch_size=16,
    emb_size=768,
    nheads=8,
    num_layers=6,
    num_outputs=len(fields)  # This now matches your label count
)

cnn = SimpleCNN()

model = vit.to(device) if False else cnn.to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
#optimizer = optim.Adam(model.parameters(), lr=lr)  # Assuming lr is defined
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Training loop
num_epochs = 10  # Adjust this as needed
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        # move to gpu (lots-of-op) / cpu (no-op)
        data, target = data.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate MAE for each label
        mae_per_label = torch.mean(torch.abs(output - target), dim=0)
        
        running_loss += loss.item()
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f'epoch [{epoch+1}/{num_epochs}], step [{batch_idx}/{len(dataloader)}], loss: {loss.item():.4f}', end='')
            for label_idx in range(len(fields)):
                print(f', mae {label_idx}: {mae_per_label[label_idx].item():.4f}', end='')
            print()  # New line for the next print statement

    # Print epoch loss
    average_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}')
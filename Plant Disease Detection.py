
# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# -----------------------------
# Parameters
# -----------------------------
EPOCHS = 10  # Reduce for Colab demo
BS = 32
IMAGE_SIZE = (256, 256)
DIRECTORY_ROOT = '../input/plantvillage/'  # Change if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------
# Dataset Class
# -----------------------------
class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# -----------------------------
# Load images and labels
# -----------------------------
image_paths, labels = [], []

print("[INFO] Loading images ...")
for plant_folder in os.listdir(DIRECTORY_ROOT):
    if plant_folder == ".DS_Store": continue
    plant_path = os.path.join(DIRECTORY_ROOT, plant_folder)
    
    for disease_folder in os.listdir(plant_path):
        if disease_folder == ".DS_Store": continue
        disease_path = os.path.join(plant_path, disease_folder)
        for img_file in os.listdir(disease_path)[:200]:  # limit for speed
            if img_file.lower().endswith(".jpg"):
                image_paths.append(os.path.join(disease_path, img_file))
                labels.append(disease_folder)

print(f"[INFO] Total images loaded: {len(image_paths)}")

# Encode labels
label_binarizer = LabelBinarizer()
labels_encoded = label_binarizer.fit_transform(labels)
pickle.dump(label_binarizer, open("label_transform.pkl", "wb"))
n_classes = labels_encoded.shape[1]
labels_class = np.argmax(labels_encoded, axis=1)  # integer labels

# -----------------------------
# Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    image_paths, labels_class, test_size=0.2, random_state=42
)

# -----------------------------
# Data augmentation & transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

train_dataset = PlantDataset(x_train, y_train, transform=train_transform)
test_dataset = PlantDataset(x_test, y_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

# -----------------------------
# CNN Model
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(3), nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*21*21, 1024),
            nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNModel(n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training
# -----------------------------
train_losses, test_losses = [], []
train_acc, test_acc = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_losses.append(running_loss / len(train_dataset))
    train_acc.append(correct / total)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_losses.append(val_loss / len(test_dataset))
    test_acc.append(correct / total)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]*100:.2f}% "
          f"Val Loss: {test_losses[-1]:.4f}, Val Acc: {test_acc[-1]*100:.2f}%")

# -----------------------------
# Plot results
# -----------------------------
plt.figure()
plt.plot(range(1, EPOCHS+1), train_acc, 'b', label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), test_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses, 'b', label='Training Loss')
plt.plot(range(1, EPOCHS+1), test_losses, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "cnn_model.pth")
print("[INFO] Model saved as cnn_model.pth")

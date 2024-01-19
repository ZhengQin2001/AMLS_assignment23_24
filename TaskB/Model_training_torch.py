import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from keras.utils import to_categorical
from torchvision import transforms
import os
from PreprocessingB import Path_Preprocessing
from PIL import Image
from torchvision.transforms import ToPILImage
import math
import time
import torch.utils.model_zoo as model_zoo



model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}

# ResNet code adapted for 9-class classification
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # Adjust the first convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Omitting the max pooling layer to preserve feature map size
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Use adaptive pooling to handle smaller feature maps
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward method adjustments
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Max pooling layer is removed
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Model_training_path_torch(nn.Module):
    def __init__(self, dataset_path="Dataset"):
        path_datapath = os.path.join(dataset_path, 'pathmnist.npz')

        try:
            path_data = np.load(path_datapath)
        except FileNotFoundError:
            print(f"Dataset file not found in path: {path_datapath}")
            return

        self.train = Path_Preprocessing(path_data, 'train', 'pathmnist')
        self.validation = Path_Preprocessing(path_data, 'val', 'pathmnist')
        self.test = Path_Preprocessing(path_data, 'test', 'pathmnist')

        self.X_train, self.y_train = self.train.normalized_images, self.train.labels
        self.X_val, self.y_val = self.validation.normalized_images, self.validation.labels
        self.X_test, self.y_test = self.test.normalized_images, self.test.labels
        
        self.train_dataset = self.create_dataset(self.X_train, self.y_train, augment=True)
        self.val_dataset = self.create_dataset(self.X_val, self.y_val)
        self.test_dataset = self.create_dataset(self.X_test, self.y_test)

    def create_dataset(self, X, y, augment=False):
        # Normalize X
        X_normalized = X / 255.0
        X_tensor = torch.tensor(X_normalized).float().permute(0, 3, 1, 2)  # Convert to [N, C, H, W] format

        if augment:
            transform = transforms.Compose([
                ToPILImage(),  # Convert tensor to PIL Image
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                # Add more transformations as needed
                transforms.ToTensor(),  # Convert back to tensor
            ])

            def apply_transform(x):
                x = transform(x)
                return x

            X_tensor = torch.stack([apply_transform(x) for x in X_tensor])

        y_tensor = torch.tensor(np.argmax(y, axis=1)).long()
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def load_resnet34_model(self, pretrained=False, num_classes=9):
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        if pretrained:
            # Load the pretrained weights
            pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
            model_dict = model.state_dict()

            # Filter out unnecessary keys from pretrained_dict (specifically the fc layer)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)

        # Replace the last layer with a new one for 9 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def train_model(self, model, epochs=10, batch_size=32, learning_rate=0.001, weight_decay=0.001, patience=3):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        best_val_accuracy = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            start_time = time.time()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print(f'Epoch: {epoch+1:03d}/{epochs:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Cost: {loss.item():.4f}')

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'Epoch: {epoch+1:03d}/{epochs:03d} | Train: {val_accuracy:.3f}%')
            print(f'Time elapsed: {time.time() - start_time:.2f} min')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping")
                break

        print('Training complete')
        print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')


    def evaluate_model(self, model, batch_size=32):
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')

    def save_model(self, model, file_name="resnet34_trained.pth"):
        torch.save(model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

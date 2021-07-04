from PIL import Image
import os
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models

image_path = '../data/images'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 512))
])

class PurePlantDataset(Dataset):
    def __init__(self, ids, labels, transform, augment = None):
        
        super(PurePlantDataset, self).__init__()
        self.cache = {}
        self.transform = transform
        self.augment = augment
        self.ids = ids
        self.labels = labels
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        image = self.cache.get(index, None)
        if image == None:
            image_name = os.path.join(image_path, self.ids[index] + '.jpg')
            image = Image.open(image_name)
            image = self.transform(image)
            self.cache[index] = image
        
        if self.augment != None:
            return self.augment(image), self.labels[index]
        return image, self.labels[index]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, X):
        return X

class ResnetModel(nn.Module):
    def __init__(self, backbone, n_class):
        super(ResnetModel, self).__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = Identity()
        
        self.fc = nn.Linear(in_features, n_class)
    def forward(self, X):
        out = self.backbone(X)
        return self.fc(out)
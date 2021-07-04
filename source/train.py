import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold

from model import *

# make data set====================================================
train_csv_path = './train.csv'
train_info = pd.read_csv(train_csv_path)
quantity = train_info.describe()
class_weight = (quantity.loc['mean'].max() / quantity.loc['mean']).values
id_images = train_info['image_id'].values
labels = train_info.iloc[:, 1:].values
labels = labels.argmax(axis = 1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 512))
])
augment = transforms.RandomApply(transforms = [transforms.GaussianBlur(11),
                                         transforms.RandomPerspective(),
                                         transforms.RandomRotation(degrees=(0, 180)),
                                         transforms.RandomAutocontrast(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip()])

# training hyperparameters ========================================
n_epoch = 20
lr = 1e-2
batch_size = 32
n_fold = 5 # test size 0.2
global_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# evaluate ========================================================
def get_accuracy(logit, y_truth):
    logit = torch.softmax(logit, dim = 1)
    y_predict = logit.argmax(dim = 1)
    return torch.sum(y_predict == y_truth) / len(y_truth)

def evaluate(model, criterion, test_loader):
    with torch.no_grad():
        losses = []
        accs = []
        for X, Y in test_loader:
            X = X.to(device = global_device)
            Y = Y.to(device = global_device)
            out = model(X)
            loss = criterion(out.detach(), Y).item() if criterion != None else -1
            acc = get_accuracy(out.detach(), Y)
            
            losses.append(loss)
            accs.append(acc.item())
        return sum(losses)/len(losses), sum(accs)/len(accs)

# train ========================================================
freeze = False
checkpoint_path = '../checkpoint/model_resnet50.pt'
continue_training = False

spliter = StratifiedKFold(n_splits=n_fold, shuffle=True)
best_score = 0
for i, (train_idx, val_idx) in enumerate(spliter.split(id_images, labels)):
    print(f"Train in fold {i}============================================")
    train_data = PurePlantDataset(ids=id_images[train_idx], 
                                    labels = labels[train_idx], 
                                    transform = transform, 
                                    augment = augment)
    val_data = PurePlantDataset(ids=id_images[val_idx], 
                                    labels = labels[val_idx], 
                                    transform = transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    model = ResnetModel('resnet50', 4)
    model.to(device = global_device)
    if freeze == True:
        for param in model.backbone.parameters():
            param.requires_grad = False

    opt_param = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.Adam(params=opt_param, lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    if os.path.exists(checkpoint_path) & continue_training:
        checkpoint = torch.load(checkpoint_path, map_location=global_device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_score = checkpoint['best_score']
        print("Load state dict")
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float, device=global_device))
    for epoch in range(n_epoch):
        print(f"Train epoch {epoch}/{n_epoch}...")
        total_loss = []
        total_acc = []
        
        model.train()
        for X, Y in tqdm(train_loader):
            optimizer.zero_grad()
            X = X.to(device = global_device)
            Y = Y.to(device = global_device)
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            
            total_loss.append(loss.item())
            acc = get_accuracy(out.detach(), Y)
            total_acc.append(acc.item())
        
        model.eval()
        avg_train_loss = sum(total_loss) / len(total_loss)
        avg_train_acc = sum(total_acc) / len(total_acc)
        val_loss, val_acc = evaluate(model, criterion, val_loader)
        print(f"Train loss {avg_train_loss:.4f} accuray {avg_train_acc:.4f}. Val loss {val_loss:.4f} accuracy {val_acc:.4f}")
        
        if val_acc >= best_score:
            best_score = val_acc
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score
            }, checkpoint_path)
            print("Save---")
        scheduler.step()
    break # train on fold 0

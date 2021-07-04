import os
import numpy as np
import pandas as pd
from model import *

global_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_pretrain(mode, path, device):
    model = ResnetModel(mode, 4)
    model.to(device=global_device)

    checkpoint = torch.load(path, map_location=device)
    model.to(device=device)
    model.load_state_dict(checkpoint['model'])
    return model
  

model = ResnetModel('resnet50', 4)
model.to(device=global_device)

checkpoint =torch.load('../checkpoint/model_resnet50.pt', map_location=global_device)
model.load_state_dict(checkpoint['model'])
model.eval()

df = pd.read_csv('test.csv')
for title in ['healthy', 'multiple_diseases', 'rust', 'scab']:
    df[title] = np.NaN
for i, id_image in enumerate(df.image_id):
    image_name = os.path.join(image_path, id_image + '.jpg')
    image = Image.open(image_name)
    image = transform(image)
    image = image.to(device = global_device)
    output = model(image.unsqueeze(0))
    logit = torch.softmax(output.detach().cpu(), dim = 1)
    
    round_logit = [round(l, 2) for l in logit.squeeze().tolist()]
    df.iloc[i, 1:] = round_logit

df.to_csv("PhamQuocHuy_supmission.csv", index=False)

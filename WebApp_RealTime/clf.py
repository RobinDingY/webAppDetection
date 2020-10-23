
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models.utils import load_state_dict_from_url
import re
from io import StringIO
from PIL import Image
from torch.autograd import Variable
import cv2

softmax_func = nn.Softmax(dim=1)

label_classes = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six',
    7:'good', 8:'none'
}

transform_test = transforms.Compose([
    transforms.Resize((224,224)),                
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


cls = models.densenet121(pretrained=False)
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])
    
class Densenet(nn.Module):
    def __init__(self,num_outputs):
        super(Densenet,self).__init__()
        self.densenet = cls

        self.densenet.fc = nn.Linear(1024, num_outputs)
          
    def forward(self,x):
        out = self.densenet(x)
        return out



def predict(skeleton_bgr):
    # device = torch.device("cpu")
    output_dim = 9
    d_net = Densenet(num_outputs = output_dim).cpu()
    #d_net = Densenet(num_outputs=output_dim).cuda()
    d_net.load_state_dict(torch.load('cleaning_densenet_97.pth',map_location=torch.device('cpu')))
    d_net.eval()
    cv_img = cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img)
    image = transform_test(pil_img).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0).cpu()
    #image = image.unsqueeze(0).cuda()

    label = np.argmax(softmax_func(d_net(image)).detach().cpu().numpy())
    label_class = label_classes[label]
    return label_class
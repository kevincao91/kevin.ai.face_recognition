from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18, resnet34, resnet50, vgg16

net = vgg16()

print(net)

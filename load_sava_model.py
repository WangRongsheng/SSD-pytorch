import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from nets.ssd import get_ssd
from nets.ssd_training import Generator, MultiBoxLoss
from utils.config import Config
from utils.dataloader import SSDDataset, ssd_dataset_collate

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#在torch 1.6版本中重新加载一下网络参数
model = get_ssd("train", Config["num_classes"]).to(device) #实例化模型并加载到cpu或GPU中
model.load_state_dict(torch.load('logs/Epoch100-Total_Loss1.5409-Val_Loss1.7362.pth', map_location=device))
#重新保存网络参数，此时注意改为非zip格式
model_cp = './logs/final.pth'
torch.save(model.state_dict(),model_cp,_use_new_zipfile_serialization=False)

#########################################################
# Import library
# --------
#
import torch 
import torch.nn as nn     
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from shutil import copyfile
import time
import numpy as np
import os
import argparse
import yaml
import collections
from tqdm import tqdm
from extra_function import *
from model import *
# Options
# Environment
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--num_workers', type=int, default=10, help='number of workers for data loader')
# data
parser.add_argument('--data_dir',default='../ReID_Dataset/market1501/pytorch',type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--DG', action='store_true', help='use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.' )
# optimizer
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
# backbone
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_swin', action='store_true', help='use swin transformer 224x224' )
parser.add_argument('--use_swinv2', action='store_true', help='use swin transformerv2' )
parser.add_argument('--use_efficient', action='store_true', help='use efficientnet-b4' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--use_hr', action='store_true', help='use hrNet' )
parser.add_argument('--use_convnext', action='store_true', help='use ConvNext' )
parser.add_argument('--ibn', action='store_true', help='use resnet+ibn' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
# loss
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
parser.add_argument('--instance', action='store_true', help='use instance loss' )
parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')
parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0' )
parser.add_argument('--aiter', default=10, type=float, help='use adv loss with iter' )

args = parser.parse_args()

data_dir = args.data_dir
name = args.name
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
args.gpu_ids = gpu_ids
# set gpu ids
if len(gpu_ids)>0:
    #torch.cuda.set_device(gpu_ids[0])
    cudnn.enabled = True
    cudnn.benchmark = True
    
    
######################################################################
# Load Data
# ---------
#
if args.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128
    
# Dataset Loader
transform_train_list = [
    transforms.Resize((h, w), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((h, w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

if args.PCB: #Using PCB+Resnet
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if args.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = args.erasing_p, mean=[0.0, 0.0, 0.0])]

if args.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)

data_transforms = {
'train': transforms.Compose(transform_train_list),
'val': transforms.Compose(transform_val_list)
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                        data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                        data_transforms['val'])
image_datasets['test'] = datasets.ImageFolder(os.path.join(args.data_dir, 'query'),
                                        data_transforms['val'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize,
                                            shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                            prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
            for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
class_num = len(class_names)

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
model = ft_net(class_num = class_num)
if len(args.gpu_ids) < 2 :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
else:
    model = torch.nn.DataParallel(model, range(len(args.gpus))).cuda() 
    
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.total_epoch*2//3, gamma=0.1)

def train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=args.total_epoch):
    since = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            # Phases 'train' and 'val' are visualized in two separate progress bars
            pbar = tqdm()
            pbar.reset(total=len(dataloaders[phase].dataset))
            ordered_dict = collections.OrderedDict(Phase="", Loss="", Acc="")

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for iter, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                pbar.update(now_batch_size)  # update the pbar even in the last batch
                #if now_batch_size<args.batch_size: # skip the last batch
                #    continue
                
                # Put model in GPU device
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                del inputs
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
                # loss.item() contains the loss of the entire mini-batch
                # It’s because the loss given loss functions is divided by the number of elements i.e. 
                # the reduction parameter is mean by default(divided by the batch size).  
                running_loss += loss.item() * now_batch_size
                
                ordered_dict["Loss"] = f"{loss.item():.4f}"
                
                del loss
                running_corrects += float(torch.sum(preds == labels.data))
                # Refresh the progress bar in every batch
                ordered_dict["Phase"] = phase
                ordered_dict["Acc"] = f"{(float(torch.sum(preds == labels.data)) / now_batch_size):.4f}"
                pbar.set_postfix(ordered_dict=ordered_dict)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            ordered_dict["Phase"] = phase
            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"
            pbar.set_postfix(ordered_dict=ordered_dict)
            pbar.close()
                
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)    
            
            # deep copy the model
            if phase == 'val' and epoch%10 == 9:
                last_model_wts = model.state_dict()
                if len(args.gpu_ids)>1:
                    save_network(model.module, epoch+1)
                else:
                    save_network(model, epoch+1)
            if phase == 'val':
                draw_curve(epoch)
            if phase == 'train':
                scheduler.step()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(last_model_wts)
    if len(args.gpu_ids)>1:
        save_network(model.module, 'last')
    else:
        save_network(model, 'last')
    
    ######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

dir_name = os.path.join('./model',name)
os.makedirs(dir_name, exist_ok=True)

# record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/args.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(args), fp, default_flow_style=False)

train_model()
# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

arch = 'alexnet'
num_classes = 365
batch_size = 40
epochs = 10000
d_input_size = 512
d_hidden_size = 200
d_output_size = 2
d_lr = 1e-6
target_lr = 1e-5
stepsize = 150

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def extract(v):
    return v.data.storage().tolist()

def adjust_lr(optimizer, epoch):
    lr = target_lr * (0.1 ** (epoch // stepsize))
    print (lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Initialize dataloader
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

source_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('sourcetest', transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=1, pin_memory=True)

target_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('targettest', transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=1, pin_memory=True)

#Load model
#model_file = 'whole_%s_places365_python36.pth.tar' % arch
model_file = 'whole_wideresnet18_places365_python36.pth.tar'
useGPU = 1
if useGPU == 1:
    model = torch.load(model_file)
else:
    model = torch.load(model_file, map_location=lambda storage, loc: storage)  # model trained in GPU could be deployed in CPU machine like this!

#define source model and target model
sourcemodel = model
sourcemodel.cuda()
print (sourcemodel)
targetmodel = copy.deepcopy(sourcemodel)
targetmodel.cuda()
#print (targetmodel)
for i, param in enumerate(targetmodel.parameters()):
    if (i >= 10): param.requires_grad = False

#for param in targetmodel.parameters():
#    print (param.requires_grad, ' ', param.size())

#define discrminator and training label
D = Discriminator(input_size=d_input_size,hidden_size= d_hidden_size, output_size=d_output_size)
D.cuda()
print (D)
source_adv_label = autograd.Variable(torch.LongTensor(batch_size).zero_()).cuda()
target_adv_label = autograd.Variable(torch.LongTensor(batch_size).zero_() + 1).cuda()
adv_label =  torch.cat((source_adv_label, target_adv_label), 0).cuda()

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
discriminator_optimizer = optim.Adam(D.parameters(), lr = d_lr)
target_optimizer = optim.Adam(filter(lambda p: p.requires_grad, targetmodel.parameters()), lr = target_lr)

#Save loss
f = open('Loss.txt', 'w')

for epoch in range(1, epochs + 1):
    if (epoch % stepsize == 0):
        torch.save(targetmodel, arch + str(epoch))
        #adjust learning rate
        #adjust_lr(optimizer=target_optimizer, epoch=epoch)

    #Train discriminator
    #Get data from 2 domain
    (source, _) = next(iter(source_loader))
    (target, _) = next(iter(target_loader))

    #Get features map of 2 domain
    s = torch.autograd.Variable(source).cuda()
    sm = sourcemodel.features(s)
    t = torch.autograd.Variable(target).cuda()
    tm = targetmodel.features(t)

    #Features for training discriminator
    adv_feat = torch.cat((sm.view(batch_size, -1), tm.view(batch_size, -1)), 0)
    #print (adv_feat.size())

    D.zero_grad()
    logits = D(adv_feat)
    #print (logits.size())
    adv_loss = criterion(logits, adv_label)
    adv_loss.backward()
    discriminator_optimizer.step()

    #Train target mapper
    targetmodel.zero_grad()
    (target, _) = next(iter(target_loader))
    t = torch.autograd.Variable(target).cuda()
    tm = targetmodel.features(t)
    tm = tm.view(batch_size, -1)
    logits = D(tm)
    #print (logits.size())
    map_loss = criterion(logits, 1 - target_adv_label)
    map_loss.backward()
    target_optimizer.step()

    f.write(str(epoch) + str(extract(adv_loss)[0]) + str(extract(map_loss)[0]) + '\n')
    print("%s: adv_loss: %s map_loss: %s" % (epoch, extract(adv_loss)[0], extract(map_loss)[0]))

f.close()

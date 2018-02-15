# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import glob
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
from torchvision import transforms as trn

arch = 'wideresnet18'
num_classes = 365
batch_size = 40
epochs = 10000
d_input_size = 512
d_hidden_size = 500
d_output_size = 2
d_lr = 1e-5
target_lr = 1e-6
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

def hook_feature(module, input, output):
    features_blobs.append(output)

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
    num_workers=4, pin_memory=True)

target_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('targettest', transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

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

#Hook function to get features
sourcemodel._modules.get('avgpool').register_forward_hook(hook_feature)
targetmodel._modules.get('avgpool').register_forward_hook(hook_feature)

for i, param in enumerate(targetmodel.parameters()):
    if (i >= 60):
        param.requires_grad = False
    #print (i, ' ', param.size())

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
f = open('Loss_adaptation.txt', 'w')

features_blobs = []
best_loss = 100000

for epoch in range(1, epochs + 1):
    #if (epoch % stepsize == 0):
    #    torch.save(targetmodel, arch + str(epoch))
        #adjust learning rate
        #adjust_lr(optimizer=target_optimizer, epoch=epoch)

    #Train discriminator
    #Get data from 2 domain
    (source, _) = next(iter(source_loader))
    (target, _) = next(iter(target_loader))

    features_blobs = []
    #Get features map of 2 domain
    s = torch.autograd.Variable(source).cuda()
    sourcemodel.forward(s)
    sm = features_blobs[0] #source features
    t = torch.autograd.Variable(target).cuda()
    targetmodel.forward(t)
    tm = features_blobs[1] #target features

    #Features for training discriminator
    adv_feat = torch.cat((sm.view(batch_size, -1), tm.view(batch_size, -1)), 0)
    #print (adv_feat.size())

    D.zero_grad()
    logits = D(adv_feat)
    #print (logits.size())
    adv_loss = criterion(logits.cuda(), adv_label)
    adv_loss.backward()
    discriminator_optimizer.step()

    #Train target mapper
    targetmodel.zero_grad()
    (target, _) = next(iter(target_loader))
    t = torch.autograd.Variable(target).cuda()
    targetmodel.forward(t)
    tm = features_blobs[2]
    tm = tm.view(batch_size, -1)
    logits = D(tm)
    map_loss = criterion(logits, 1 - target_adv_label)
    map_loss.backward()
    target_optimizer.step()

    if (epoch % 10 == 0):
        f.write(str(epoch) + ' ' + str(extract(adv_loss)[0]) + ' ' + str(extract(map_loss)[0]) + '\n')
        print("%s: adv_loss: %s map_loss: %s " % (epoch, extract(adv_loss)[0], extract(map_loss)[0]))

        centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class10 = ['barn', 'beach', 'bedroom', 'castle', 'classroom', 'desert', 'kitchen', 'library', 'mountain', 'river']

        target_loss = 0
        s = ''
        countim = 0
        for c in class10:
            imgs = glob.glob('targettest/' + c + '/*.jpg')
            countim += len(imgs)
            #print (len(imgs))
            class_loss = 0
            ct = 0
            for cli in range(0, 365):
                if (classes[cli] == c or (classes[cli] == 'desert/sand' and c in classes[cli]) or (classes[cli] == 'library/indoor' and c in classes[cli])):
                    ct = cli
                    break
            for img_name in imgs:
                img = Image.open(img_name)
                input_img = torch.autograd.Variable(centre_crop(img).unsqueeze(0), volatile=True).cuda()
                logit = targetmodel.forward(input_img)
                label = autograd.Variable(torch.LongTensor(1).zero_() + ct).cuda()
                loss = criterion(logit, label)
                class_loss = class_loss + extract(loss)[0]
            print (c, ': ', len(imgs))
            s = s + str(class_loss/len(imgs)) + ' '
            target_loss = target_loss + class_loss
        print (countim)
        f.write(s + '\n')
        target_loss = target_loss / countim
        f.write(str(target_loss) + '\n')
        print('loss_per_class: ' + s)
        print('target_loss: ', target_loss)
        if (target_loss < best_loss):
            best_loss = target_loss
            torch.save(targetmodel, arch + str(epoch))

        #f.write(str(epoch) + ' ' + str(extract(adv_loss)[0]) + ' ' + str(extract(map_loss)[0]) + ' ' + str(source_loss) + ' ' + str(target_loss) + '\n')
        #print("%s: adv_loss: %s map_loss: %s source_loss: %s target_loss: %s" % (epoch, extract(adv_loss)[0], extract(map_loss)[0], source_loss, target_loss))


f.close()

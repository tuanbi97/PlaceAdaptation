# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import os
import glob
import numpy as np
from PIL import Image

# th architecture to use
arch = 'wideresnet18'
#model_file = 'whole_wideresnet18_places365_python36.pth.tar'
model_file = 'wideresnet182730'
save_file = 'Loss_classifier_2730'
batch_size = 10
epochs = 10000
lr = 1e-3
stepsize = 100
c_input = 512
c_hidden_size = 200
c_output_size = 10

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x


def hook_feature(module, input, output):
    features_blobs.append(output)

# load the pre-trained weights
useGPU = 1
if useGPU == 1:
    model = torch.load(model_file)
else:
    model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

## assume all the script in python36, so the following is not necessary
## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
#from functools import partial
#import pickle
#pickle.load = partial(pickle.load, encoding="latin1")
#pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
#torch.save(model, 'whole_%s_places365_python36.pth.tar'%arch)

model.cuda()
model._modules.get('avgpool').register_forward_hook(hook_feature)
print (model)

model.eval()

# load the image transformer
data_transforms = {
	'targettest_70': trn.Compose([
		trn.RandomSizedCrop(224),
		trn.RandomHorizontalFlip(),
        trn.ToTensor(),
		trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]),
	'targettest_30': trn.Compose([
		trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
}

data_dir = './'
images_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
				   for x in ['targettest_70', 'targettest_30']}
dataloaders = {x: torch.utils.data.DataLoader(images_datasets[x], batch_size = 40, shuffle = True, num_workers = 2)
			   for x in ['targettest_70', 'targettest_30']}

dataset_sizes = {x: len(images_datasets[x]) for x in ['targettest_70', 'targettest_30']}
classes = images_datasets['targettest_70'].classes
print(classes)

#define classifier
classifier = Classifier(c_input, c_hidden_size, c_output_size)
classifier.cuda()

#define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr = lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
best_acc = 0
best_loss = 1000000

features_blobs = []
fout = open(save_file, 'w')

for epoch in range(1, epochs + 1):
    print ('Epoch: ', epoch)
    s = 'Epoch: ' + str(epoch)
    for phase in ['targettest_70', 'targettest_30']:
        if phase == 'targettest_70':
            #scheduler.step()
            classifier.train(True)
        else:
            classifier.train(False)
        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders[phase]:
            features_blobs = []
            inputs, labels = data
            inputs = V(inputs.cuda())
            labels = V(labels.cuda())
            optimizer.zero_grad()

            model.forward(inputs)
            feats = features_blobs[0].view(inputs.size(0), -1)
            outputs = classifier(feats)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            if (phase == 'targettest_70'):
                loss.backward()
                optimizer.step()

            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        s = s + ' {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
        if (phase == 'targettest_30'):
            if((epoch_acc > best_acc) or (epoch_acc == best_acc and epoch_loss < best_loss)):
                best_acc = epoch_acc
                best_loss = epoch_loss
                torch.save(classifier, 'classifier' + str(epoch))

    fout.write(s + '\n')
fout.close()

# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import glob
import numpy as np
from PIL import Image

# th architecture to use
arch = 'wideresnet18'
#model_file = 'wideresnet181750'
model_file = 'whole_wideresnet18_places365_python36.pth.tar'
classifier_file = 'classifier22_origin'
#save_file = 'TargetAlexNetAdaptation9600.csv'
save_file = 'TargetResNet18_origin_classifier_train.csv'
class10 = ['barn', 'beach', 'bedroom', 'castle', 'classroom', 'desert', 'kitchen', 'library', 'mountain', 'river']
class30 = ['airport_terminal', 'apartment_building_outdoor' ,'arch', 'auditorium', 'conference_room', 'dam', 
            'football_stadium', 'great_pyramid', 'hotel_room', 'office', 'phone_booth', 'reception', 'restaurant',
            'school_house', 'shower', 'skyscraper', 'supermarket', 'waiting_room', 'water_tower', 'windmill',
            'barn', 'beach', 'bedroom', 'castle', 'classroom', 'desert', 'kitchen', 'library', 'mountain', 'river']
id_classes30 = {'airport_terminal': 2, 'apartment_building_outdoor': 8,'arch': 12, 'auditorium': 27, 'conference_room': 102, 'dam': 113, 
            'football_stadium': 313, 'great_pyramid': 116, 'hotel_room': 182, 'office': 244, 'phone_booth': 263, 'reception': 280, 'restaurant' : 284,
            'school_house': 296, 'shower': 303, 'skyscraper': 307, 'supermarket': 321, 'waiting_room': 352, 'water_tower': 354, 'windmill': 361,
            'barn': 40, 'beach': 48, 'bedroom': 52, 'castle': 84, 'classroom': 92, 'desert': 116, 'kitchen': 203, 'library': 212, 'mountain': 232, 'river': 288}

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

if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

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

classifier = torch.load(classifier_file)
print (classifier)
classifier.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the test image
imgs = []
for c in class30:
	imgs = imgs + glob.glob('targettest_70/' + c + '/*.jpg')
fout = open(save_file, 'w') 
s = 'image_name'
for c in class30:
    s = s + ',' + c
s = s + ',top 1'
fout.write(s + '\n')

features_blobs = []

for img_name in imgs:
    features_blobs = []
    img = Image.open(img_name)
    #print (np.array(img).shape)
    input_img = V(centre_crop(img).unsqueeze(0), volatile=True).cuda()

    # forward pass
    model.forward(input_img)
    #print (len(features_blobs))
    feats = features_blobs[0].view(1, -1)
    logit = classifier(feats)
    h_x = F.softmax(logit, 1).data.squeeze()

    s = img_name
    for i in range(0, 30):
        s = s + ',{:.3f}'.format(h_x[i])

    probs, idx = h_x.sort(0, True)

    print('RESULT ON ' + img_name)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], class30[idx[i]]))
    s = s + ',' + class30[idx[0]]
    fout.write(s +'\n')

fout.close()

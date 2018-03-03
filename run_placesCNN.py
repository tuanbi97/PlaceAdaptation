# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import glob
import numpy as np
from PIL import Image

# th architecture to use
arch = 'wideresnet18'
model_file = 'wideresnet182730'
#model_file = 'whole_wideresnet18_places365_python36.pth.tar'
#save_file = 'TargetAlexNetAdaptation9600.csv'
save_file = 'TargetResNet182730.csv'
class10 = ['barn', 'beach', 'bedroom', 'castle', 'classroom', 'desert', 'kitchen', 'library', 'mountain', 'river']

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
print (model)

model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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

# load the test image
imgs = []
for c in class10:
	imgs = imgs + glob.glob('targettest/' + c + '/*.jpg')
fout = open(save_file, 'w') 
s = 'image_name'
for c in class10:
	s = s + ',' + c
s = s + ',top 1'
fout.write(s + '\n')

features_blobs = []

for img_name in imgs:
	img = Image.open(img_name)
	#print (np.array(img).shape)
	input_img = V(centre_crop(img).unsqueeze(0), volatile=True).cuda()

	# forward pass
	logit = model.forward(input_img)
	h_x = F.softmax(logit, 1).data.squeeze()

	s = img_name
	for i in range(0, len(classes)):
		if ((classes[i] in class10) or (classes[i] == 'desert/sand') or (classes[i] == 'library/indoor')):
			s = s + ',{:.3f}'.format(h_x[i])

	probs, idx = h_x.sort(0, True)

	print('RESULT ON ' + img_name)
	for i in range(0, 5):
		print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
	s = s + ',' + classes[idx[0]]
	fout.write(s +'\n')

fout.close()

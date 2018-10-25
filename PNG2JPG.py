from PIL import Image
import os
import glob

dirs = os.listdir('./targettest_70')
print(len(dirs))
print(dirs)

for i in range(0, len(dirs)):
	image_folder = dirs[i]
	image_paths = glob.glob('./targettest_70/' + image_folder +'/*.PNG')
	for image_path in image_paths:
		print(image_path)
		im = Image.open(image_path)
		im.load()
		background = Image.new("RGB", im.size, (255, 255, 255))
		background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
		background.save(image_path[0:-4] + '.jpg')

# ========================================================
# To display feature maps at different layers.
# --------------------------------------------------------
# Sangeeta Biswas, Ph.D.
# Associate Professor,
# Dept. of CSE, University of Rajshahi,
# Rajshahi-6205, Bangladesh.
# sangeeta.cse.ru@gmail.com / sangeeta.cse@ru.ac.bd
# -------------------------------------------------------
# 13/11/2025
# =======================================================

#--- Import necessary modules from Python libraries.
from tensorflow.keras.applications import vgg16, mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, pickle

#--- Fixed terms
WORKING_DIR = '/home/bibrity/Research/CV2024/'  
IMG_SIZE = 224

def main():
	#--- Prepare image
	img_path = WORKING_DIR + 'puppy_cat.jpeg'
	img = cv2.imread(img_path) #--- Load BGR image
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #--- Convert BGR image into RGB image
	print(img.shape, img.dtype, img.max(), img.min())
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #--- Resize image
	print(img.shape, img.dtype, img.max(), img.min())
	img = np.expand_dims(img, 0) #--- Turn 3D image into 4D data for Conv2D layers
	print(img.shape, img.dtype, img.max(), img.min())
	img = vgg16.preprocess_input(img) #--- Preprocess image according to the steps followed by the pre-trained model
	print(img.shape, img.dtype, img.max(), img.min())
	
	#--- Load a pre-trained backbone
	base_model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (IMG_SIZE, IMG_SIZE, 3))
	base_model.summary(show_trainable = True)
	
	#--- Build model
	output_layer_number = 10
	inputs = base_model.input
	outputs = base_model.layers[output_layer_number].output
	model = Model(inputs, outputs)
	
	#--- Display feature maps
	feature_mapset = model.predict(img)
	print(feature_mapset.shape)
	img_set = []
	for i in range(9):
		img_set.append(feature_mapset[0, :, :, i])
	plot_images(img_set, row = 3, col = 3)

def plot_images(img_set, title_set = '', row = 1, col = 1, fig_path = ''):
	n = len(img_set)
	plt.rcParams.update({'font.size': 16})
	plt.figure(figsize = (10, 10))
	for i in range(n):
		plt.subplot(row, col, i + 1)
		img = img_set[i]
		if (len(img.shape) == 3): 
			ch = img.shape[-1]
			if (ch == 1): #--- For 3D grayscale image
				plt.imshow(img[:, :, 0], cmap = 'gray')
			elif (ch == 3): #--- For 3D RGB image or 3D one-hot encoded image
				plt.imshow(img)
		else: #--- For 2D grayscale image.
			plt.imshow(img, cmap = 'gray')
		
		plt.axis('off')
		if (title_set != ''):
			plt.title(title_set[i])

	if (fig_path != ''):
		plt.savefig(fig_path)
	else:
		plt.show()
	plt.close()

if __name__ == '__main__':
	main()

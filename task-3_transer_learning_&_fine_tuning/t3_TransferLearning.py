# ========================================================
# To train and test a classifier using Transfer Learning.
# --------------------------------------------------------
# https://keras.io/guides/transfer_learning/
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
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications import vgg16, mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, pickle

#--- Fixed terms
# WORKING_DIR = '/home/bibrity/Research/CV2024/'  
WORKING_DIR = '/home/mursalin/m3c/computer-vision/check/'  
IMG_SIZE = 32
EARLY_STOP_PATIENCE = 20
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.8 #--- new_lr = old_lr * LR_REDUCE_FACTOR
NUM_CLASSES = 10
WARMUP_EPOCHS = 2
EPOCHS = 3

def main():
	#--- Create a directory to store model and figures
	storage_dir = WORKING_DIR + 'TenClass_Classification/' 
	if (os.path.exists(storage_dir) == False):
		os.makedirs(storage_dir)
	else:
		print(storage_dir + ' exists.')
	
	#--- Prepare data
	(trainX, trainY), (testX, testY) = process_data()
	
	#--- Train a classifier using Transfer learning
	train_classifier(storage_dir, trainX, trainY)
	
	#--- Test trained classifier
	test_classifier(storage_dir, testX, testY)
	
def test_classifier(storage_dir, testX, testY):
	#--- Load trained model
	model = build_model()	
	model_path = storage_dir + 'VGG16_Classifier.weights.h5'
	model_weights = model.load_weights(model_path)
	
	#--- Compile model when we need metrics not mentioned while training
	model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy', Precision()])
	
	#--- Predict model's output
	predictedY = np.argmax(model.predict(testX), axis = -1)
	int_testY = np.argmax(testY, axis = -1)
	n = predictedY.shape[0]
	print('Original_Y 	Predicted_Y')
	print('========== 	===========')	
	for i in range(n):
		print('{}       {}'.format(int_testY[i], predictedY[i]))
		
	#--- Evaluate model performance
	model.evaluate(testX, testY)
		
def train_classifier(storage_dir, trainX, trainY):
	#--- Build model
	model = build_model()
	model.summary(show_trainable = True)
	
	#--- Freez backbone
	for layer in model.layers[:-5]:
		layer.trainable = False
	model.summary(show_trainable = True)
	
	#--- Train model
	model.fit(trainX, trainY, validation_split = 0.2, epochs = WARMUP_EPOCHS) #--- Warm-up training
	
	#--- Unfreez some Convolutional layers of backbone for fine-tuning
	for layer in model.layers[-7:-5]:
		layer.trainable = True
	model.summary(show_trainable = True)	
	
	#--- Callbacks
	model_path = storage_dir + 'VGG16_Classifier.weights.h5'
	callbacks = [
		ModelCheckpoint(model_path, monitor = "val_loss", mode = 'min', save_best_only = True, save_weights_only = True),
		EarlyStopping(monitor = "val_loss", mode = 'min', patience = EARLY_STOP_PATIENCE),
		ReduceLROnPlateau(monitor = "val_loss", mode = 'min', factor = LR_REDUCE_FACTOR, patience = LR_REDUCE_PATIENCE)
	]

	#--- Train model
	hist = model.fit(trainX, trainY, validation_split = 0.2, epochs = EPOCHS, callbacks = callbacks) #--- Fine-tuning
	
	#--- Save history
	performance_path = storage_dir + 'TrainVal_'
	save_model_performance(performance_path, hist)
	
def save_model_performance(performance_path, history):
	#--- Save history into a dictionary
	hist_dict = history.history
	with open(performance_path + 'PerformanceDict.pkl', 'wb') as f:
		pickle.dump(hist_dict, f)

	#--- Plot progress graphs
	x_axis = np.arange(len(hist_dict['loss']))
	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize = (20, 20))
	plt.plot(x_axis, hist_dict['loss'], 'k.--', linewidth = 2, markersize = 12)
	plt.plot(x_axis, hist_dict['val_loss'], 'g*--', linewidth = 2, markersize = 12)
	plt.xticks(rotation = 90)
	plt.legend(['training_loss', 'validation_loss'])
	plt.savefig(performance_path + 'Loss.jpg')
	plt.close()

	metric = 'accuracy'
	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize = (20, 20))
	plt.plot(x_axis, hist_dict[metric], 'k.--', linewidth = 2, markersize = 12)
	plt.plot(x_axis, hist_dict['val_' + metric], 'g*--', linewidth = 2, markersize = 12)
	plt.xticks(rotation = 90)
	plt.legend(['training_' + metric, 'validation_' + metric])
	plt.savefig(performance_path + metric + '.jpg')
	plt.close()

def process_data():
	#-- Load data
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data() 
	
	#--- Turn 3D image dataset into 4D dataset for Conv2D layers
	print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
	print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))
	trainX = convert_3D_to_4D(trainX)
	testX = convert_3D_to_4D(testX)	 
	print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
	print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))
	
	#--- Preprocess imageset according to the preprocess procedure of pre-trained model
	trainX = vgg16.preprocess_input(trainX)
	testX = vgg16.preprocess_input(testX)
	print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
	print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))
			
	#--- Turn y as one-hot-encoding
	print('trainY.shape: {}, trainY.dtype: {}'.format(trainY.shape, trainY.dtype))
	print('testY.shape: {}, testY.dtype: {}'.format(testY.shape, testY.dtype))
	trainY = to_categorical(trainY, NUM_CLASSES)
	testY = to_categorical(testY, NUM_CLASSES)
	print('trainY.shape: {}, trainY.dtype: {}'.format(trainY.shape, trainY.dtype))
	print('testY.shape: {}, testY.dtype: {}'.format(testY.shape, testY.dtype))
		
	#--- Cross check
	# plt.imshow(trainX[0])
	# plt.title(trainY[0])
	# plt.show()
	# plt.close()
	
	return (trainX, trainY), (testX, testY)
	
def convert_3D_to_4D(x):
	n, h, w = x.shape
	x4D = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype = np.uint8)
	for i in range(n):
		#--- Resize image
		resized_img = cv2.resize(x[i], (IMG_SIZE, IMG_SIZE))
		
		#--- Convert 2D image into 3D image
		x4D[i] = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) 
	return x4D
	
def build_model():
	#--- Load a pre-trained backbone
	base_model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (IMG_SIZE, IMG_SIZE, 3))
	base_model.summary(show_trainable = True)
		
	#--- Build a new model based on loaded backbone
	inputs = base_model.input
	x = base_model.output
	x = layers.Flatten()(x)
	x = layers.Dense(128, activation = 'relu')(x)
	x = layers.Dense(64, activation = 'relu')(x)	
	outputs = layers.Dense(10, activation = 'softmax')(x)
	model = Model(inputs, outputs)
	
	#--- Compile model
	model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	return model
	
if __name__ == '__main__':
	main()

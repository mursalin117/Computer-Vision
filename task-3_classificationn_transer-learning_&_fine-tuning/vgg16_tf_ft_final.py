# ========================================================
# To train and test a classifier using Transfer Learning.
# =======================================================

#--- Import necessary modules from Python libraries.
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import vgg16, mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, pickle
import csv

#--- Fixed terms
WORKING_DIR = '/home/mursalin/m3c/computer-vision/check/'  
IMG_SIZE = 32
EARLY_STOP_PATIENCE = 20
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.8 #--- new_lr = old_lr * LR_REDUCE_FACTOR
NUM_CLASSES = 10
WARMUP_EPOCHS = 2
EPOCHS = 3
RUN_NO = 1

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
    history = train_classifier(storage_dir, trainX, trainY)
        
    #--- Test trained classifier
    test_metrics = test_classifier(storage_dir, testX, testY)

    # Define CSV file and column headers
    csv_file =  storage_dir + "multiple_runs_metrics.csv"
    fieldnames = ["run", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "test_loss", "test_accuracy"]

    # Get training and validation metrics from the last epoch
    train_loss = history.history["loss"][-1]
    train_accuracy = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    test_loss = test_metrics[0]
    test_accuracy = test_metrics[1]
        
    # Append metrics to CSV file
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "run": RUN_NO,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })


def test_classifier(storage_dir, testX, testY):
    #--- Load trained model
    model = build_model()	
    model_path = storage_dir + 'VGG16_Classifier.weights.h5'
    model_weights = model.load_weights(model_path)
    
    #--- Compile model when we need metrics not mentioned while training
    model.compile(loss = 'categorical_crossentropy', metrics = [Accuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
    
    #--- Predict model's output
    predictedY = np.argmax(model.predict(testX), axis = -1)
    int_testY = np.argmax(testY, axis = -1)
    # n = predictedY.shape[0]
    n = 10
    print('Original_Y 	Predicted_Y')
    print('========== 	===========')	
    for i in range(n):
        print('{}                 {}'.format(int_testY[i], predictedY[i]))
    
    #--- Evaluate model performance
    test_metrics = model.evaluate(testX, testY)
    # print(test_metrics)

    # for csv file saving 
    # test_metrics = dict(zip(model.metrics_names, test_metrics))
    # print(test_metrics)
    return test_metrics
        
        
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
    
    return hist


def save_model_performance(performance_path, history):
    #--- Save history into a dictionary
    hist_dict = history.history
    with open(performance_path + 'PerformanceDict.pkl', 'wb') as f:
        pickle.dump(hist_dict, f)

    #--- Plot progress graphs
    # Plot loss
    x_axis = np.arange(len(hist_dict['loss']))
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (20, 20))
    plt.plot(x_axis, hist_dict['loss'], 'k.--', linewidth = 2, markersize = 12)
    plt.plot(x_axis, hist_dict['val_loss'], 'g*--', linewidth = 2, markersize = 12)
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.title('Training and Validation Loss')
    plt.xticks(rotation = 90)
    plt.legend(['training_loss', 'validation_loss'])
    plt.savefig(performance_path + 'Loss.jpg')
    plt.close()

    # Plot accuracy
    metric = 'accuracy'
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (20, 20))
    plt.plot(x_axis, hist_dict[metric], 'k.--', linewidth = 2, markersize = 12)
    plt.plot(x_axis, hist_dict['val_' + metric], 'g*--', linewidth = 2, markersize = 12)
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.title('Training and Validation Accuracy')
    plt.xticks(rotation = 90)
    plt.legend(['training_' + metric, 'validation_' + metric])
    plt.savefig(performance_path + metric + '.jpg')
    plt.close()

    # Plot precision
    metric = 'precision'
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (20, 20))
    plt.plot(x_axis, hist_dict[metric], 'k.--', linewidth = 2, markersize = 12)
    plt.plot(x_axis, hist_dict['val_' + metric], 'g*--', linewidth = 2, markersize = 12)
    plt.xlabel('Precision')
    plt.ylabel('Epoch')
    plt.title('Training and Validation Precision')
    plt.xticks(rotation = 90)
    plt.legend(['training_' + metric, 'validation_' + metric])
    plt.savefig(performance_path + metric + '.jpg')
    plt.close()

    # Plot recall
    metric = 'recall'
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (20, 20))
    plt.plot(x_axis, hist_dict[metric], 'k.--', linewidth = 2, markersize = 12)
    plt.plot(x_axis, hist_dict['val_' + metric], 'g*--', linewidth = 2, markersize = 12)
    plt.xlabel('Recall')
    plt.ylabel('Epoch')
    plt.title('Training and Validation Recall')
    plt.xticks(rotation = 90)
    plt.legend(['training_' + metric, 'validation_' + metric])
    plt.savefig(performance_path + metric + '.jpg')
    plt.close()

    # Plot auc
    metric = 'auc'
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (20, 20))
    plt.plot(x_axis, hist_dict[metric], 'k.--', linewidth = 2, markersize = 12)
    plt.plot(x_axis, hist_dict['val_' + metric], 'g*--', linewidth = 2, markersize = 12)
    plt.xlabel('auc')
    plt.ylabel('Epoch')
    plt.title('Training and Validation AUC')
    plt.xticks(rotation = 90)
    plt.legend(['training_' + metric, 'validation_' + metric])
    plt.savefig(performance_path + metric + '.jpg')
    plt.close()

    # # Plot Precision, Recall, and AUC
    # metrics = ['precision', 'recall', 'auc']
    # plt.figure(figsize=(20, 20))
    # for idx, metric in enumerate(metrics):
    # 	plt.subplot(1, 3, idx + 1)
    # 	plt.plot(hist_dict[metric], label=metric.capitalize())
    # 	plt.plot(hist_dict[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
    # 	plt.xlabel('Epoch')
    # 	plt.ylabel(metric.capitalize())
    # 	plt.title(f'Training and Validation {metric.capitalize()}')
    # 	plt.xticks(rotation = 90)
    # 	plt.legend()
    # plt.tight_layout()
    # plt.savefig(performance_path + 'Precision-Recall-AUC' + '.jpg')
    # # plt.show()
    # plt.close()


def process_data():
    #-- Load data
    # (trainX, trainY), (testX, testY) = fashion_mnist.load_data() 
    (trainX, trainY), (testX, testY) = cifar10.load_data() 
    
    #--- Turn 3D image dataset into 4D dataset for Conv2D layers
    print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
    print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))
    # no need for cifar10
    # trainX = convert_3D_to_4D(trainX)
    # testX = convert_3D_to_4D(testX)	 
    # print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
    # print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))
    
    # normalize data -> no need when preprocessing function is used 
    # trainX = trainX / 255.0
    # testX = testX / 255.0
    # print('trainX.shape: {}, trainX.dtype: {}'.format(trainX.shape, trainX.dtype))
    # print('testX.shape: {}, testX.dtype: {}'.format(testX.shape, testX.dtype))


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
    

# def convert_3D_to_4D(x):
# 	n, h, w = x.shape
# 	x4D = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype = np.uint8)
# 	for i in range(n):
# 		#--- Resize image
# 		resized_img = cv2.resize(x[i], (IMG_SIZE, IMG_SIZE))
        
# 		#--- Convert 2D image into 3D image
# 		x4D[i] = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) 
# 	return x4D


# def resize_images(images):
#     # Resize images to (224, 224, 3) as required by the VGG16 model
#     resized_images = np.zeros((images.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
#     for i in range(images.shape[0]):
#         resized_images[i] = cv2.resize(images[i], (IMG_SIZE, IMG_SIZE))
#     return resized_images

    
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
    model.compile(loss = 'categorical_crossentropy', metrics = [Accuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
    
    return model
    
if __name__ == '__main__':
    main()


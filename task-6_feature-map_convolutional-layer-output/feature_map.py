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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import vgg16, mobilenet
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, pickle

#--- Fixed terms
WORKING_DIR = '/home/mursalin/m3c/computer-vision/task/feature-map-1/'  
IMG_SIZE = 224

def main():
    #--- Prepare image
    # img_path = WORKING_DIR + 'puppy_cat.jpeg'
    # img = cv2.imread(img_path) #--- Load BGR image
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    img = testX[1] #--- Load BGR image	
    print(img.shape, img.dtype, img.max(), img.min())
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #--- Resize image
    print(img.shape, img.dtype, img.max(), img.min())
    img = np.expand_dims(img, 0) #--- Turn 3D image into 4D data for Conv2D layers
    print(img.shape, img.dtype, img.max(), img.min())
    # img = vgg16.preprocess_input(img) #--- Preprocess image according to the steps followed by the pre-trained model
    img = mobilenet.preprocess_input(img) #--- Preprocess image according to the steps followed by the pre-trained model
    print(img.shape, img.dtype, img.max(), img.min())
    
    #--- Load a pre-trained backbone
    # base_model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (IMG_SIZE, IMG_SIZE, 3))
    base_model = mobilenet.MobileNet(include_top = False, weights = 'imagenet', input_shape = (IMG_SIZE, IMG_SIZE, 3))
    base_model.summary(show_trainable = True)

    for layer in range(1, len(base_model.layers)):
        #--- Build model
        output_layer_number = layer
        inputs = base_model.input
        outputs = base_model.layers[output_layer_number].output
        model = Model(inputs, outputs)
        
        #--- Display feature maps
        feature_mapset = model.predict(img)
        print(feature_mapset.shape)
        img_set = []
        img_set.append(testX[1])
        
        for i in range(1, 9):
            img_set.append(feature_mapset[0, :, :, i])
        plot_images(img_set, row = 3, col = 3, fig_path = WORKING_DIR, layer = layer)

def plot_images(img_set, title_set = '', row = 1, col = 1, fig_path = '', layer = 0):
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
        fig_path = fig_path + 'layer-' + str(layer) + '.jpg'
        plt.title('Layer-' + str(layer))
        plt.savefig(fig_path)
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    main()

def train_classifier(storage_dir, trainX, trainY, batch_size):
    #--- Build model
    model = build_model()
    model.summary(show_trainable = True)
    
    #--- Freez backbone
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.summary(show_trainable = True)

    # Split the training data into training and validation sets
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2)
    
    # Data generator for batching
    datagen = ImageDataGenerator()
    # batch_size = 8  # Start with a small batch size
    steps_per_epoch = len(trainX) // batch_size
    
    # Train the model with data generator
    model.fit(
        datagen.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=(valX, valY),
        epochs=5
    )
    
    #--- Unfreez some Convolutional layers of backbone for fine-tuning
    for layer in model.layers[-7:-5]:
        layer.trainable = True
    model.summary(show_trainable = True)	
    
    #--- Callbacks
    # model_path = storage_dir + 'VGG16_Classifier.weights.h5'
    model_path = storage_dir + 'VGG16_Classifier.weights.keras'
    callbacks = [
        ModelCheckpoint(model_path, monitor = "val_loss", mode = 'min', save_best_only = True, save_weights_only = False),
        EarlyStopping(monitor = "val_loss", mode = 'min', patience = EARLY_STOP_PATIENCE),
        ReduceLROnPlateau(monitor = "val_loss", mode = 'min', factor = LR_REDUCE_FACTOR, patience = LR_REDUCE_PATIENCE)
    ]


    # Train the model with data generator
    hist = model.fit(
        datagen.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=(valX, valY),
        epochs=5, 
        callbacks=callbacks
    )
    
    #--- Save history
    performance_path = storage_dir + 'TrainVal_'
    save_model_performance(performance_path, hist)
    
    return hist
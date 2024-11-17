# Import the necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset():
    # Load and preprocess CIFAR-10 dataset
    # (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train_full.shape)

    # normalize values of pixel
    x_train_full = x_train_full / 255.0
    x_test = x_test / 255.0

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
    print(x_train.shape)

    x_train = x_train.reshape((x_train.shape[0], 28, 28))
    x_val = x_val.reshape((x_val.shape[0], 28, 28))
    x_test = x_test.reshape((x_test.shape[0], 28, 28))
    print(x_train.shape)

    # convert to 3d image
    x_train = np.stack((x_train,) * 3, axis=-1)
    x_val = np.stack((x_val,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)
    print(x_train.shape)

    # Resize to (224, 224, 3) -> for high resources
    #x_train = tf.image.resize(x_train, (224, 224)).numpy()
    #x_test = tf.image.resize(x_test, (224, 224)).numpy()
    
    # Resize each image to (128, 128, 3) individually
    x_train = np.array([tf.image.resize(img, (128, 128)).numpy() for img in x_train])
    x_val = np.array([tf.image.resize(img, (128, 128)).numpy() for img in x_val])
    x_test = np.array([tf.image.resize(img, (128, 128)).numpy() for img in x_test])
    print(x_train.shape)

    # Preprocess data by vgg16.preprocess_input() function
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    x_test = preprocess_input(x_test)

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# Create datasets
(trainX, trainY), (valX, valY), (testX, testY) = create_dataset()

# Define VGG16 model with transfer learning
def create_model():
    # Load VGG16 without the top classification layers, use input shape (128, 128, 3)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.summary()
    print('--------------------------')

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create a new top layer
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes for Fashion MNIST
    
    return model

# Compile the model
model = create_model()
model.summary()
print('--------------------------')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#print(len(train_dataset))
print('--------------------------')
model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(valX, valY))

# Fine-tune some layers (optional)
# Unfreeze specific layers to train them further
for layer in model.layers[0].layers[-4:]:  # Unfreezing last 4 layers of VGG16
    layer.trainable = True

model.summary()
print('--------------------------')
# Recompile the model after unfreezing
#model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
#model.fit(train_dataset, epochs=5, batch_size=32, validation_data=test_dataset)


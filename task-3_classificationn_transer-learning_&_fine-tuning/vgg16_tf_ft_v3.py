import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#
# # Normalize the data
# x_train, x_test = x_train / 255.0, x_test / 255.0
# # Convert labels to one-hot encoding
# y_train_one_hot = to_categorical(y_train, 10)
# y_test_one_hot = to_categorical(y_test, 10)
# # Function to preprocess each image
# def preprocess_image(image, label):
#
# Resize and convert to 3-channel (RGB) image
# image = tf.image.resize(image[..., tf.newaxis], (224, 224))
# image = tf.repeat(image, 3, axis=-1)

# # Apply VGG16 preprocessing
# image = preprocess_input(image)
# return image, label

# # Create a tf.data.Dataset for training and testing
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
# train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
# test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
print(x_train.shape)

# x_train = x_train.reshape((x_train.shape[0], 28, 28))
# x_test = x_test.reshape((x_test.shape[0], 28, 28))
# print(x_train.shape)
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = np.stack((x_train,) * 3, axis=-1)
# x_test = np.stack((x_test,) * 3, axis=-1)
# print(x_train.shape)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# train_dataset = (x_train, y_train)
# test_dataset = (x_test, y_test)

# # Reshaping the data to (28, 28)
x_train = x_train.reshape((x_train.shape[0], 28, 28))
x_test = x_test.reshape((x_test.shape[0], 28, 28))
print(f"Original train shape: {x_train.shape}")
# Resize the images to (224, 224)
# x_train = tf.image.resize(x_train[..., np.newaxis], (224, 224)) # Adding an extra dimens
# x_test = tf.image.resize(x_test[..., np.newaxis], (224, 224))
x_train = tf.image.resize(x_train[..., np.newaxis], (64, 64)) # Reduce image size to (128,
x_test = tf.image.resize(x_test[..., np.newaxis], (64, 64))
# Normalize to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
# Convert grayscale images to RGB by stacking the same image across 3 channels
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
print(f"Resized train shape: {x_train.shape}")
# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Create dataset tuples
train_dataset = (x_train, y_train)
test_dataset = (x_test, y_test)

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
# Add custom classification layers on top of VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
# Create a new modelmodel = Model(inputs=base_model.input, outputs=outputs)
model.summary()

# Freeze base model layers for warm training
for layer in base_model.layers:
    layer.trainable = False
model.summary(show_trainable = True)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy')

# Train the model (warm training phase)
# model.fit(train_dataset,epochs=30, validation_data=test_dataset)
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

# Fine-tuning: Unfreeze some layers and continue training
for layer in base_model.layers[-5:]: # Unfreeze last few layers
    layer.trainable = True
model.summary(show_trainable = True)

# Compile the model again after unfreezing layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy')

# Fine-tune the model
# model.fit(train dataset, epochs=20, validation data=test dataset)

# Fine-tuning: Unfreeze some layers and continue training
for layer in base_model.layers[-5:]: # Unfreeze last few layers
    layer.trainable = True 
model.summary(show_trainable = True)

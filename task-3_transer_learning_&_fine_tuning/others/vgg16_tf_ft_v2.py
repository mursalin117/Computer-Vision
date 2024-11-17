import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load and preprocess the Fashion MNIST dataset
def process_data():
    # Load Fashion MNIST dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    
    # Expand dimensions for grayscale channel and resize images to (224, 224, 3)
    trainX = tf.image.resize(tf.expand_dims(trainX, axis=-1), [224, 224])
    testX = tf.image.resize(tf.expand_dims(testX, axis=-1), [224, 224])
    
    # Convert grayscale to RGB (duplicate channels)
    trainX = tf.image.grayscale_to_rgb(trainX)
    testX = tf.image.grayscale_to_rgb(testX)

    # Preprocess images using VGG16's preprocess_input
    trainX = preprocess_input(trainX / 255.0)  # Ensure values are scaled correctly
    testX = preprocess_input(testX / 255.0)
    
    # One-hot encode labels
    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10)
    
    return (trainX, trainY), (testX, testY)


# Load preprocessed data
(trainX, trainY), (testX, testY) = process_data()

# Define VGG16 model with transfer learning
def create_model():
    # Load VGG16 without the top classification layers, use input shape (224, 224, 3)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))

# Fine-tune some layers (optional)
# Unfreeze specific layers to train them further
for layer in model.layers[0].layers[-4:]:  # Unfreezing last 4 layers of VGG16
    layer.trainable = True

# Recompile the model after unfreezing
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY))

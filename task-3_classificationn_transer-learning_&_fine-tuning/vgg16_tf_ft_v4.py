from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def main():
    #--- Prepare data
    (trainX, trainY), (testX, testY) = process_data()

    #--- Build model
    model = build_model()
    model.summary()

    #--- Freeze backbone
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.summary()

    #--- Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), customized_metric_function])

    #--- Warm-up training
    hist_wt = model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=32)

    #--- Unfreeze some layers for fine-tuning
    for layer in model.layers[-7:-5]:
        layer.trainable = True
    model.summary()

    #--- Compile again with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), customized_metric_function])

    #--- Fine-tuning
    hist_ft = model.fit(trainX, trainY, validation_split=0.2, epochs=30, batch_size=32)

    #--- Test model
    predictedY = model.predict(testX)
    model.evaluate(testX, testY)


def customized_metric_function(y_true, y_pred):
    # Example of a customized metric: categorical cross-entropy
    return tf.keras.metrics.categorical_crossentropy(y_true, y_pred)


def build_model():
    #--- Load a pretrained backbone
    base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    #--- Freeze backbone layers
    for layer in base_model.layers:
        layer.trainable = False

    #--- Build a new model based on loaded backbone
    inputs = base_model.input
    x = layers.Resizing(224, 224)(inputs)          # Resize layer for CIFAR images
    x = vgg16.preprocess_input(x)                   # Preprocess layer for VGG16
    x = base_model(x, training=False)               # Pass through VGG16
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)

    #--- Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def myprintFile(model, file_name):
    with open(file_name, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def process_data():
    #--- Load data    
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print(f"Train shape: {trainX.shape}, Test shape: {testX.shape}")

    #--- Preprocess data
    # Resize data to 224x224x3
    trainX = tf.image.resize(trainX[..., tf.newaxis], [224, 224]) / 255.0
    testX = tf.image.resize(testX[..., tf.newaxis], [224, 224]) / 255.0

    # Convert grayscale images to RGB (3 channels)
    trainX = tf.image.grayscale_to_rgb(trainX)
    testX = tf.image.grayscale_to_rgb(testX)

    # Preprocess data by vgg16.preprocess_input() function
    trainX = vgg16.preprocess_input(trainX)
    testX = vgg16.preprocess_input(testX)

    # One-hot encode labels
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)

    # Visualize an example
    plt.imshow(trainX[0].numpy().astype("uint8"))
    plt.title(f"Label: {trainY[0].argmax()}")
    plt.show()
    plt.close()

    return (trainX, trainY), (testX, testY)


if __name__ == '__main__':
    main()

from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import layers
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

    #--- Compile model with additional metrics
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    #--- Warm-up training with frozen layers
    hist_wt = model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=32)

    #--- Unfreeze some layers for fine-tuning
    for layer in model.layers[-7:-5]:
        layer.trainable = True
    model.summary()

    #--- Compile again with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    #--- Fine-tuning with the unfrozen layers
    hist_ft = model.fit(trainX, trainY, validation_split=0.2, epochs=30, batch_size=32)

    #--- Test model
    model.evaluate(testX, testY)

    #--- Plot training history
    plot_training_history(hist_wt, hist_ft)


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

    return model


# def process_data():
#     #--- Load data    
#     (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
#     print(f"Train shape: {trainX.shape}, Test shape: {testX.shape}")

#     #--- Preprocess data
#     # Expand dimensions to add a channel (grayscale -> 1 channel)
#     trainX = tf.expand_dims(trainX, axis=-1)
#     testX = tf.expand_dims(testX, axis=-1)
    
#     # Convert grayscale images to RGB (3 channels)
#     trainX = tf.image.grayscale_to_rgb(trainX)
#     testX = tf.image.grayscale_to_rgb(testX)

#     # Resize images to 224x224 as required by VGG16
#     trainX = tf.image.resize(trainX, [224, 224]) / 255.0
#     testX = tf.image.resize(testX, [224, 224]) / 255.0

#     # Preprocess data using VGG16 preprocessing
#     trainX = vgg16.preprocess_input(trainX)
#     testX = vgg16.preprocess_input(testX)

#     # One-hot encode labels
#     trainY = to_categorical(trainY, 10)
#     testY = to_categorical(testY, 10)

#     # Visualize an example
#     plt.imshow(trainX[0].numpy().astype("uint8"))
#     plt.title(f"Label: {trainY[0].argmax()}")
#     plt.show()
#     plt.close()

#     return (trainX, trainY), (testX, testY)



def process_data():
    #--- Load data    
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print(f"Train shape: {trainX.shape}, Test shape: {testX.shape}")

    #--- Preprocess data
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


def plot_training_history(hist_wt, hist_ft):
    # Combine histories from both training phases
    history = {}
    for key in hist_wt.history.keys():
        history[key] = hist_wt.history[key] + hist_ft.history[key]
        
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Precision, Recall, and AUC
    metrics = ['precision', 'recall', 'auc']
    plt.figure(figsize=(12, 4))
    for idx, metric in enumerate(metrics):
        plt.subplot(1, 3, idx + 1)
        plt.plot(history[metric], label=metric.capitalize())
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

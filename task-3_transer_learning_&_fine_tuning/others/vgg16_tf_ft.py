from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision
import numpy as np


def main():
    #--- Prepare data
    (trainX, trainY), (testX, testY) = process_data()
    
    #--- Build model
    model = build_model()
    model.summary(show_trainable=True)

    #--- Freeze backbone
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.summary(show_trainable=True)

    #--- Compile model for initial training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision()])
    
    #--- Train model (warm-up training)
    hist_wt = model.fit(trainX, trainY, validation_split=0.2, epochs=5, batch_size=32)

    #--- Unfreeze last few layers of the backbone for fine-tuning
    for layer in model.layers[-7:-5]:
        layer.trainable = True
    model.summary(show_trainable=True)

    #--- Compile model for fine-tuning
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5), metrics=['accuracy', Precision()])
    
    #--- Fine-tune model
    hist_ft = model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=32)

    #--- Plot training and validation accuracy
    plot_training_history(hist_wt, hist_ft)

    #--- Test model
    model.evaluate(testX, testY)


def customized_metric_function(y_true, y_pred):
    # Example: calculate F1-score
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def build_model():
    #--- Load a pretrained backbone
    base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    #--- Build a new model based on loaded backbone
    x = layers.InputLayer(input_shape=(28,28,3))
    x = layers.Resizing(224, 224, interpolation="bilinear")(x)
    x = vgg16.preprocess_input(x)
    inputs = base_model.input(x)
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  
    model = Model(inputs, outputs)

    #--- Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model    


def myprintFile(model, file_name):
    with open(file_name, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def process_data():
    #--- Load data    
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    #--- Preprocess data
    #expand 1 more dimention as 1 for colour channel gray
    trainX = trainX.reshape(trainX.shape[0], 28, 28,1)
    testX = testX.reshape(testX.shape[0], 28, 28,1)

    # Convert the images into 3 channels
    trainX = np.repeat(trainX, 3, axis=3)
    testX = np.repeat(testX, 3, axis=3)
    
    # One-hot encode labels
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)

    # Visualize an example
    plt.imshow(trainX[0].astype("uint8"))
    plt.title(f"Label: {trainY[0].argmax()}")
    plt.show()
    plt.close()

    return (trainX, trainY), (testX, testY)


def plot_training_history(hist_wt, hist_ft):
    # Combine histories
    acc = hist_wt.history['accuracy'] + hist_ft.history['accuracy']
    val_acc = hist_wt.history['val_accuracy'] + hist_ft.history['val_accuracy']
    loss = hist_wt.history['loss'] + hist_ft.history['loss']
    val_loss = hist_wt.history['val_loss'] + hist_ft.history['val_loss']

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

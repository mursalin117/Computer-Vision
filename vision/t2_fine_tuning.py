from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision


def main():
    #--- Prepare data
    # (trainX, trainY), (testX, testY) = process_data()
    
    #--- Build model
    model = build_model()
    model.summary(show_trainable = True)

    #--- Freeze backbone
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.summary(show_trainable = True)

    #--- Train model 
    # hist_wt = model.fit(trainX, trainY, validation_split = 0.2, epochs = 30) #--- Warm-up training -> less epoch

    #--- Freeze backbone
    for layer in model.layers[-7:-5]:
        layer.trainable = True
    model.summary(show_trainable = True)

    #--- Train model 
    # hist_ft = model.fit(trainX, trainY, validation_split = 0.2, epochs = 300) #--- Fine tuning -> higher epoch

    #--- Compile model -> when we need metrics not mentioned before training
    # model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy', Precision(), customized_metric_function])
    
    #--- Test model
    # predictedY = model.predict(testX)
    # model.evaluate(textX, testY)    


def customized_metric_function(y_true, y_pred):
    # implement the function
    return something


def build_model():
    #--- Load a pretrained backbone
    base_model = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
    base_model.summary(show_trainable = True)

    # #--- Freeze backbone
    # for layer in base_model.layers:
    #     layer.trainable = False

    #--- Build a new model based on loaded backbone
    inputs = base_model.input
    # resize_layer
    # preprocess_layer
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dense(64, activation = 'relu')(x)
    outputs = layers.Dense(10, activation = 'softmax')(x)  
    model = Model(inputs, outputs)

    # myprintFile(vgg16_model, 'vgg16.txt')
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True)

    #--- Compile model
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model    


def myprintFile(model, file_name):
    with open(file_name, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def process_data():
    #--- Load data    
    # (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # print(trainX.shape, trainX.dtype)
    # print(testX.shape, testX.dtype)

    #--- Preprocess data
    # Resize data
    # Preprocess data by vgg16.preprocess_input() function
    # turn y as one-hot-encoding 


    #--- Cross 
    plt.imshow(trainX[0])
    plt.title(trainY[0])
    plt.show()
    plt.close()

    return (trainX, trainY), (testX, testY)


if __name__ == '__main__':
    main()
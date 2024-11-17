from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.utils import plot_model

def main():
    # (trainX, trainY), (testX, testY) = process_data()
    mobilenet_model = mobilenet.MobileNet()
    mobilenet_model.summary(show_trainable = True)
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    myprintFile(mobilenet_model, 'mobilenet_full.txt')
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
    # plot_model(mobilenet_model, 'MobileNet_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=400, show_layer_activations=True, show_trainable=True)
    plot_model(mobilenet_model, 'MobileNet_full.png')
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='LR', expand_nested=True, show_layer_activations=True)
    
    mobilenet_model = mobilenet.MobileNet(include_top = False)
    mobilenet_model.summary(show_trainable = True)
    myprintFile(mobilenet_model, 'mobilenet_backbone.txt')
    plot_model(mobilenet_model, 'MobileNet_backbone.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)

def myprintFile(model, file_name):
    with open(file_name, 'w') as f:
        model.summary(show_trainable = True, print_fn=lambda x: f.write(x + '\n'))

# def myprint(s):
#     with open('modelsummary.txt','a') as f:
#         print(s, file=f)

def process_data():
    #-- Load data    
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print(trainX.shape, trainX.dtype)
    print(testX.shape, testX.dtype)

    # --- Preprocess data
    # ---
    # ---

    # --- Cross 
    plt.imshow(trainX[0])
    plt.title(trainY[0])
    plt.show()
    plt.close()

    return (trainX, trainY), (testX, testY)

if __name__ == '__main__':
    main()
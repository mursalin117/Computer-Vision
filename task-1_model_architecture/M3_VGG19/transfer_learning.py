from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.utils import plot_model

def main():
    # (trainX, trainY), (testX, testY) = process_data()
    vgg19_model = vgg19.VGG19()
    vgg19_model.summary(show_trainable = True)
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    myprintFile(vgg19_model, 'vgg19_full.txt')
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
    plot_model(vgg19_model, 'VGG19_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='LR', expand_nested=True, show_layer_activations=True)
    
    vgg19_model = vgg19.VGG19(include_top = False)
    vgg19_model.summary(show_trainable = True)
    myprintFile(vgg19_model, 'vgg19_backbone.txt')
    plot_model(vgg19_model, 'VGG19_backbone.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)

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
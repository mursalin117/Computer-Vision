from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.applications import resnet50
from tensorflow.keras.utils import plot_model

def main():
    # (trainX, trainY), (testX, testY) = process_data()
    resnet50_model = resnet50.ResNet50()
    resnet50_model.summary(show_trainable = True)
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    myprintFile(resnet50_model, 'resnet50_full.txt')
    # vgg16_model.summary(show_trainable = True, print_fn=myprint)
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
    plot_model(resnet50_model, 'ResNet50_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
    # plot_model(vgg16_model, 'VGG16_full.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='LR', expand_nested=True, show_layer_activations=True)
    
    resnet50_model = resnet50.ResNet50(include_top = False)
    resnet50_model.summary(show_trainable = True)
    myprintFile(resnet50_model, 'resnet50_backbone.txt')
    plot_model(resnet50_model, 'ResNet50_backbone.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)

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
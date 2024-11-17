import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# Load CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split the full training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Normalize pixel values to between 0 and 1
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# Define the VGG16 model with the pretrained weights, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze all the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom head for CIFAR-10 (10 classes)
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Initial Training (only the head, with base model frozen)
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)

# Unfreeze some of the last layers of the base model for fine-tuning
# Example: Unfreeze the last 4 convolutional layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile the model again with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model (train both the head and unfrozen base model layers)
fine_tune_history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

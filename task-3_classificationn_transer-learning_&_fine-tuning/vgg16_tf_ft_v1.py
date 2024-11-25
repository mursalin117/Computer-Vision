import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

# Load and preprocess CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Define the base model (VGG16) and freeze its layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add a new head for CIFAR-10 classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling instead of flattening
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(10, activation="softmax")(x)  # 10 classes in CIFAR-10

# Combine base model and new head
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model with frozen layers in the base model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[-4:]:  # Adjust the number of layers to unfreeze as needed
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fine-tune the model on the training and validation sets
history_fine_tune = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

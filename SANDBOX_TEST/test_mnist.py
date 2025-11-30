import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Display initial shapes
print("Initial x_train shape:", x_train.shape)
print("Initial y_train shape:", y_train.shape)
print("Initial x_test shape:", x_test.shape)
print("Initial y_test shape:", y_test.shape)

# Reshape data to include channel dimension (for grayscale images, it's 1)
# Keras expects input shape (batch_size, height, width, channels)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Display processed shapes
print("\nProcessed x_train shape:", x_train.shape)
print("Processed y_train shape:", y_train.shape)
print("Processed x_test shape:", x_test.shape)
print("Processed y_test shape:", y_test.shape)

# 2. Build the CNN model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# 3. Compile and train the model
batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# 4. Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("\nTest loss:", score[0])
print("Test accuracy:", score[1])

# 5. Save the trained model
model_filename = "mnist_cnn_model.keras"
model.save(model_filename)
print(f"\nModel saved to {model_filename}")

# 6. Load the saved model
loaded_model = keras.models.load_model(model_filename)
print(f"Model loaded from {model_filename}")

# 7. Use the loaded model to make predictions (optional)
predictions = loaded_model.predict(x_test[:5])
print("\nPredictions for the first 5 test samples:")
for i, pred in enumerate(predictions):
    print(f"Sample {i}: Predicted digit {np.argmax(pred)}, True digit {np.argmax(y_test[i])}")
    
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (scale pixel values to range 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to fit CNN input format (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Function to predict a digit from an image
def predict_digit(image):
    image = image.reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape
    prediction = model.predict(image)
    return np.argmax(prediction)

# Test the model on a random image
index = np.random.randint(0, len(x_test))
test_image = x_test[index].reshape(28, 28)
predicted_label = predict_digit(test_image)

# Display the image and prediction
plt.imshow(test_image, cmap='gray')
plt.title(f'Predicted: {predicted_label}')
plt.show()

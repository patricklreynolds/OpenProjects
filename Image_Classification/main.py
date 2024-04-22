# Import necessary libraries
# OpenCV for image processing, NumPy for numerical operations
# Matplotlib for visualization, Keras for building the neural network
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load the CIFAR-10 dataset and split into training and testing sets
# Normalize the image pixel values to be between 0 and 1 for training stability
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define the class names for CIFAR-10 for easy reference
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck' ]

# Display a grid of 16 images with their corresponding labels
for i in range(16):
    plt.subplot(4, 4, i + 1) # Creates a subplot for each image
    plt.xticks([]) # Removes the x-ticks
    plt.yticks([]) # Removes the y-ticks
    plt.imshow(training_images[i], cmap=plt.cm.binary) # Display image in grayscale
    plt.xlabel(class_names[training_labels[i][0]]) # Set the label beneath each image

plt.show() # Display the figure with the images

# Reducing the dataset size for faster training in this example
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Building the CNN model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # First convolutional layer
model.add(layers.MaxPooling2D((2, 2))) # First pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Second convolutional layer
model.add(layers.MaxPooling2D((2, 2))) # Second pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Third convolutional layer
model.add(layers.Flatten()) # Flattening the 3D output to 1D for the dense layers
model.add(layers.Dense(64, activation='relu')) # Dense layer for learning the representations
model.add(layers.Dense(10, activation='softmax')) # Output layer with softmax for classification

# Compiling the model with the optimizer, loss function, and metrics to monitor
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model with the training data and also validating using the test data
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('/Users/pat-home/Generative AI Shakespeare Project/Image Classification/image_classifier.h5')

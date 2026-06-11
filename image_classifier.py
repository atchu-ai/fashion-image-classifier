import tensorflow as tf

# Fashion MNIST dataset load
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=5)

# Test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest Accuracy:", test_acc)
model.save("fashion_model.h5")

print("Model Saved Successfully!")
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle Boot']

prediction = model.predict(test_images)

print("\nPredicted Class:", class_names[prediction[0].argmax()])
print("Actual Class:", class_names[test_labels[0]])
import matplotlib.pyplot as plt

plt.imshow(test_images[0], cmap="gray")
plt.title("Predicted: " + class_names[prediction[0].argmax()])
plt.show()
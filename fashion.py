import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist

# To install libaries run: pip install numpy matplotlib tensorflow

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

loss, acc = model.evaluate(test_images, test_labels)

print(acc)
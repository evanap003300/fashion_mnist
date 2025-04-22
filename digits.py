import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


# To install libaries run: pip install numpy matplotlib tensorflow

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc) # 95%
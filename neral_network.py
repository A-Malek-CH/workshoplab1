import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train = tf.keras.utils.normalize(x_train, axis=-1)
x_test = tf.keras.utils.normalize(x_test, axis=-1)

# Model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=12)

# Save the model
model.save('workshop.h5')  # Using .h5 extension

# Reload and evaluate the model
model = tf.keras.models.load_model('workshop.h5')
loss, acc = model.evaluate(x_test, y_test)

print('loss:', loss)
print('accuracy:', acc)


model = tf.keras.models.load_model('workshop.h5')
image_number = 1
while os.path.isfile(f'digits/{image_number}.png'):
    try:
        image = cv2.imread(f'digits/{image_number}.png')[:,:,0]
        image = np.invert(np.array([image])) # Reshape the image for prediction
        prediction = model.predict(image)
        print(f"This digit is probably a {np.argmax(prediction)}") # Reshape the image for visualization
        image_reshaped = image.reshape(28, 28)
        plt.imshow(image_reshaped, cmap=plt.cm.binary)
        plt.show()
    except Exception as e: print(f"Error: {e}")
    finally: image_number +=1

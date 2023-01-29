import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

from dropPositions import getDropPositions

apiKey = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiIxYTY1NDUzMC02MjBmLTAxM2ItNGRiOS0wMzFjNzRhOWU0NTciLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNjcxNDg0MDYzLCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6ImJyaXRpc2hmYWxjb25nIn0.4vwHivcvKIG_XWyCvXfgilPDyDItaHE7ZhsahPdgf3I"
platform = "pc-eu"
match_id = "52a5331a-6290-48ff-8849-015ab3879b8a"

def plot_prediction(epoch, logs):
    aircraftPath = [0, 6181.8675, 8000, 4096.524089946661]
    start_x, start_y, end_x, end_y = aircraftPath

    test = np.array(aircraftPath, dtype=np.float32)
    test = np.expand_dims(test, axis=0)

    # Feature scaling on everything
    sc = StandardScaler()
    test = sc.fit_transform(test)

    # Test the model on the first game
    prediction = model.predict(test)
    prediction = prediction.reshape(80, 80)

    # Don't draw the map, as it slows down the training

    plt.imshow(prediction, extent=[0, 8000, 0, 8000], alpha=0.8, interpolation='bilinear', cmap='plasma')

    plt.plot([start_x, end_x], [8000-start_y, 8000-end_y], color='red', linewidth=2)

    plt.gca().invert_yaxis()

    plt.show(block=False) # Block is false so that the training can continue

    plt.pause(0.001) # Pause required to not cause a freeze in MPL when the next epoch is run

with open('oldgames.csv', 'r') as csvfile:
    # Create a CSV reader
    reader = csv.reader(csvfile)

    # Skip the header row
    next(reader)

    # Initialize arrays to store the input and output data
    x_data = []
    y_data = []

    # Iterate over the rows in the file
    for game in reader:
        # Convert the input and output data to NumPy arrays
        x = np.array(game[:4], dtype=np.float32)
        y = np.array(game[4:], dtype=np.float32)

        # Add another dimension to the input and output data
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # Append the input and output data to the arrays
        x_data.append(x)
        y_data.append(y)

# Split the data into a training set and a test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

# Concatenate the input and output data for all rows
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Feature scaling on everything
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train.shape)

num_classes = 6400
test = 256

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

plot_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_prediction)

# Train the model
model.fit(x_train, y_train, epochs=16, batch_size=32, validation_data=(x_test, y_test), callbacks=[plot_callback])

#aircraftPath, segments = getDropPositions("ea41a995-eeda-4559-8660-b308451da411", "pc-eu", apiKey)

# Keeping this here to minimise API calls for the same match
aircraftPath = [0, 6181.8675, 8000, 4096.524089946661]
start_x, start_y, end_x, end_y = aircraftPath

test = np.array(aircraftPath, dtype=np.float32)
test = np.expand_dims(test, axis=0)

# Feature scaling on everything
sc = StandardScaler()
test = sc.fit_transform(test)

# Test the model on the first game
prediction = model.predict(test)
prediction = prediction.reshape(80, 80)

# Add the Erangel map as a background
plt.imshow(plt.imread("maps/Erangel_Main_High_Res_Inverted.png"), extent=[0, 8000, 0, 8000])

plt.imshow(prediction, extent=[0, 8000, 0, 8000], alpha=0.8, interpolation='bilinear', cmap='plasma')

plt.plot([start_x, end_x], [8000-start_y, 8000-end_y], color='red', linewidth=2)

plt.gca().invert_yaxis()

plt.show()
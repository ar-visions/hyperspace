import json
import os
import numpy             as np
import tensorflow        as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers, initializers, Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from einops import *
from einops.layers.tensorflow import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import ViTModel, ViTConfig


# Check if TensorFlow is built with GPU support
print("Built with GPU support:", tf.test.is_built_with_cuda())

# Check GPUs available to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

# Define the path to the directory containing the JSON files
W = 256
H = 144
lr = 0.00001

gen_path = './gen'

fields  = ['x','y','z','rx','ry','rz','fov']
labels  = []
sources = []
i=0

p = os.getcwd()

if os.path.exists('build'):
    os.chdir('build')

# iterate over the files in the directory
for f in os.listdir(gen_path):
    if not f.endswith('.json'): continue
    count = 0
    with open(f'{gen_path}/{f}', 'r', encoding='utf-8') as fdata:
        try:
            js = json.load(fdata)
        except Exception as e:
            print(f"error occurred with file {f}: {e} (skipping...)")
            continue

        lv = []
        for field in fields:
            v = js['labels'][field]
            assert(v >= -1024 and v <= 1024)
            lv.append(v)
        
        source = f'{gen_path}/{js["source"]}'
        if not os.path.exists(source):
            continue
        assert(os.path.exists(source))
        labels.append(lv)
        sources.append(source)
        i=i+1
        if(i % 100 == 0):
            print(len(sources))

# Convert labels to a numpy array
labels = np.array(labels)

print(labels.shape) # (2200, 7)
print(len(sources)) #  2200

# Preprocess images and load them
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    #image = tf.image.resize(image, [H, W])
    image = tf.keras.applications.efficientnet.preprocess_input(image)  # Replace with appropriate preprocessing for your model
    return image

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(sources, labels, test_size=0.1, random_state=22)

# Create TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.map(lambda x, y: (load_and_preprocess_image(x), y))

test_data  = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_data  = test_data.map(lambda x, y: (load_and_preprocess_image(x), y))

# Batch and shuffle
batch_size = 1
train_data = train_data.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data  = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# data looks fine, but doesnt train even with constant numbers lol
# maybe we should try PyTorch?

"""
# Define the index of the image you want to retrieve
desired_index = 10  # for example, the 11th image in the dataset

# Iterate over the dataset to retrieve the specific image and label
for i, (image, label) in enumerate(train_data.unbatch()):  # unbatch if the dataset is batched
    if i == desired_index:
        # This is the desired image
        raw_image = image.numpy()  # Convert the TensorFlow tensor to a numpy array
        raw_label = label.numpy()  # Convert the TensorFlow tensor to a numpy array
        break

# Now raw_image contains the image tensor and raw_label contains the corresponding label

# If you need to visualize the image, use matplotlib or another library
import matplotlib.pyplot as plt

plt.imshow(raw_image.astype('uint8'))  # Make sure the image type is correct for visualization
plt.title(f'Label: {raw_label}')
plt.show()
"""


# define a custom metric for each field
def create_custom_metric(index, field_name):
    def custom_metric(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, index] - y_pred[:, index]), axis=-1)
    custom_metric.__name__ = 'mae_' + field_name
    return custom_metric

# custom metrics for each field
custom_metrics = [create_custom_metric(i, field) for i, field in enumerate(fields)]

# Number of output neurons needed (one per regression task)
num_outputs = y_train.shape[1]


def build_model(input_shape, output_shape):
    model = Sequential([
        # First convolutional layer with max pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional layer with max pooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional layer with max pooling
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten the 3D output to 1D
        Flatten(),
        
        # Dense layer with 128 units
        Dense(128, activation='relu'),
        
        # Output layer with linear activation
        Dense(output_shape, activation=None)  # No activation function (linear output)
    ])
    
    return model

# Input shape (height, width, channels)
input_shape = (H, W, 3)

# Output shape (number of regression targets)
output_shape = len(fields)

# Build the model
model = build_model(input_shape, output_shape)

# Compile the model
model.compile(optimizer=Adam(learning_rate=lr),
              loss='mean_squared_error',  # For regression tasks, mean squared error is commonly used.
              metrics=custom_metrics)  # Mean absolute error as an additional metric

# Summary of the model
model.summary()

# Train the model
history = model.fit(train_data,
                    epochs=10,  # You can decide the number of epochs
                    validation_data=test_data)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_data)
print(f"Test loss: {test_loss}")
print(f"Test MAE: {test_mae}")

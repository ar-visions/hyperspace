import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras        import layers
from tensorflow.keras.models import Model
from   sklearn.model_selection import train_test_split

from   PIL import Image
import numpy as np
import json
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs Available: ", gpus)
else:
    print("No GPUs were found")

# Define the path to the directory containing the JSON files
W = 256
H = 256
lr = 0.0001

gen_path = './gen'
fields  = ['qx', 'qy', 'qz', 'qw']
labels  = []
sources = []

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

# Convert labels to a numpy array
labels = np.array(labels)

print(labels.shape) # (2200, 8)
print(len(sources)) #  2200

# Preprocess images and load them
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.ensure_shape(image, (H, W, 3))
   #image = tf.image.resize(image, [H, W])  # Resize the image
    image = tf.cast(image, tf.float32) / 255.0  # Convert to float32 and scale pixel values
    image = tf.keras.applications.efficientnet.preprocess_input(image)
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

# Define a custom metric for each field
def create_custom_metric(index, field_name):
    def custom_metric(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, index] - y_pred[:, index]), axis=-1)
    custom_metric.__name__ = 'mae_' + field_name
    return custom_metric

# Custom metrics for each field
custom_metrics = [create_custom_metric(i, field) for i, field in enumerate(fields)]

# Define the input shape
input_shape = (256, 256, 3)


## i need a regular CNN here.  actual tensorflow model not keras

# Define the CNN model
class SimpleCNNModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.d2 = tf.keras.layers.Dense(units=len(fields))  # Assuming 'fields' is the number of outputs

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Instantiate the model
model = SimpleCNNModel()

# Compile the model with an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Define a function to calculate MSE for each component
def calculate_mse_per_field(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast labels to float32 if necessary
    y_pred = tf.cast(y_pred, tf.float32)  # Ensure predictions are float32
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=0)  # Computes MSE along the batch axis for each field

# Define the training step
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.keras.losses.MSE(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(10):
    total_mse_per_field = np.zeros(len(fields))  # Initialize MSE accumulation for each field
    for step, (images, labels) in enumerate(train_data):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = tf.keras.losses.MSE(labels, predictions)
            mse_per_field = calculate_mse_per_field(labels, predictions)  # MSE for each field
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Accumulate MSE for averaging later
        total_mse_per_field += mse_per_field.numpy()
        
        # Optionally, print the step and loss
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")

    # Average MSE for each field over all steps
    average_mse_per_field = total_mse_per_field / len(train_data)
    
    # Output the average MSE for each field
    print(f"Epoch {epoch} Average MSE per field:")
    for i, field_name in enumerate(fields):
        print(f"{field_name}: {average_mse_per_field[i]}")

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
              loss='mean_squared_error', 
              metrics=custom_metrics)


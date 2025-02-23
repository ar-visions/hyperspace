#!/usr/bin/env python3
# pip install tensorflow[gpu]
# ----------------------------------------------------------
import tensorflow as tf
from   tensorflow.keras import layers, Model
import numpy as np
import os
import json
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

size = 128  # Crop size
full_size = 340  # Original image size

class VisionDataset:
    def __init__(self, sessions_dir, validation_split=0.1):
        self.sessions_dir = sessions_dir

        self.images = []
        self.labels = []
        self.crops = []  # Store crop positions (-0.5 to 0.5)

        for filename in os.listdir(sessions_dir):
            if filename.startswith('f2_') and filename.endswith('.png'):
                parts = filename.replace('.png', '').split('_')
                if len(parts) == 7:
                    _, _, eye_x, eye_y, head_x, head_y, head_z = parts

                    # Get corresponding JSON label file
                    json_path = os.path.join(sessions_dir, filename.replace('.png', '.json'))
                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            json_data = json.load(f)
                            crop_pos = json_data["labels"][0]["eye-center"]  # Extract [-0.5, 0.5] crop position
                    else:
                        print('found single file with no annotation')
                        exit(2)
                        crop_pos = [0.0, 0.0]  # Default to center if missing

                    self.images.append(os.path.join(sessions_dir, filename))
                    self.labels.append([float(eye_x), float(eye_y)])
                    self.crops.append(crop_pos)  # Store the crop position

        # Shuffle data before splitting
        combined = list(zip(self.images, self.labels, self.crops))
        random.shuffle(combined)
        self.images, self.labels, self.crops = map(list, zip(*combined))

        # Split into train and validation
        split_index = int(len(self.images) * (1 - validation_split))
        self.train_images, self.val_images = self.images[:split_index], self.images[split_index:]
        self.train_labels, self.val_labels = self.labels[:split_index], self.labels[split_index:]
        self.train_crops, self.val_crops = self.crops[:split_index], self.crops[split_index:]


    def center_crop(self, image, crop_offset):
        """Dynamically crops based on user-provided targeting (-0.5 to 0.5 range)."""
        crop_x = tf.cast((crop_offset[0] + 0.5) * full_size, tf.int32)  # Convert -0.5 to 0.5 range into pixel values
        crop_y = tf.cast((crop_offset[1] + 0.5) * full_size, tf.int32)

        # Ensure crop is within valid bounds
        start_x = tf.clip_by_value(crop_x - size // 2, 0, full_size - size)
        start_y = tf.clip_by_value(crop_y - size // 2, 0, full_size - size)

        return tf.image.crop_to_bounding_box(image, start_y, start_x, size, size)

    def process_path(self, file_path, label, crop_offset):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img, channels=1)
        img = self.center_crop(img, crop_offset)
        img = tf.cast(img, tf.float32) / 255.0
        
        true_crop_offset = tf.convert_to_tensor(crop_offset, dtype=tf.float32)
        #forced_crop_offset = tf.constant([0.0, 0.0], dtype=tf.float32)

        return img, true_crop_offset, label

    def create_dataset(self, batch_size=1, train=True):
        images, labels, crops = (self.train_images, self.train_labels, self.train_crops) if train else (
            self.val_images, self.val_labels, self.val_crops)
        
        dataset = tf.data.Dataset.from_tensor_slices((images, crops, labels))
        dataset = dataset.map(self.process_path)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        return dataset

# keras model
def create_model():
    image_input = layers.Input(shape=(size, size, 1))  # 128x128 cropped image
    crop_input = layers.Input(shape=(2,))  # (-0.5, 0.5) crop position input

    # Conv blocks
    x = layers.Conv2D(32, 3, padding='same')(image_input)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)

    # Merge crop position with the dense layer
    merged = layers.Concatenate()([x, crop_input])
    merged = layers.Dense(512, activation='relu')(merged)

    outputs = layers.Dense(2)(merged)  # Linear output for X, Y coordinates

    model = Model(inputs=[image_input, crop_input], outputs=outputs)
    return model

# Initialize dataset
id = 'wkjykq'
dataset = VisionDataset(f'sessions/{id}')

train_ds = dataset.create_dataset(batch_size=1, train=True)
val_ds = dataset.create_dataset(batch_size=1, train=False)

# Create model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.00001),
    loss='mse',
    metrics=['mae']
)

# Custom training loop
@tf.function
def train_step(images, crop_offsets, labels):
    with tf.GradientTape() as tape:
        predictions = model([images, crop_offsets], training=True)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

@tf.function
def val_step(images, crop_offsets, labels):
    predictions = model([images, crop_offsets], training=False)
    loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
    return loss, predictions

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0

    for images, crop_offsets, labels in train_ds:
        loss, outputs = train_step(images, crop_offsets, labels)
        total_loss += loss
        total_samples += 1
        
        errors = tf.abs(outputs - labels)
        avg_errors = tf.reduce_mean(errors, axis=0)

        loss_value = float(loss.numpy())
        error_values = avg_errors.numpy()

        print(f'Epoch [{epoch+1}/{num_epochs}] Step [{total_samples}], '
              f'Loss: {loss_value:.4f} | Error: X={error_values[0]:.4f}, Y={error_values[1]:.4f}')

    avg_loss = float(total_loss.numpy()) / total_samples

    # Validation Step
    val_loss = 0
    val_samples = 0
    total_val_errors = tf.zeros(2, dtype=tf.float32)

    for val_images, val_crop_offsets, val_labels in val_ds:
        loss, val_outputs = val_step(val_images, val_crop_offsets, val_labels)
        val_loss += loss
        val_samples += val_labels.shape[0]  # Track total samples
        total_val_errors += tf.reduce_sum(tf.abs(val_outputs - val_labels), axis=0)  # Sum over batch

    # Compute **true average error** after summing over all validation samples
    avg_val_loss = float(val_loss.numpy()) / val_samples
    avg_val_errors = total_val_errors.numpy() / val_samples  # Normalize by total samples

    print(f'--- Epoch [{epoch+1}/{num_epochs}] Completed ---')
    print(f'Avg Training Loss: {avg_loss:.4f} | Avg Validation Loss: {avg_val_loss:.4f}')
    print(f'True Validation Error: X={avg_val_errors[0]:.4f}, Y={avg_val_errors[1]:.4f}')
    print('------------------------------------------\n')


# Save the trained model
model.save('vision_model.keras')

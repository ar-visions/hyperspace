#!/bin/env python3
import tensorflow as tf
from   tensorflow.keras              import KerasTensor
from   tensorflow.keras.layers       import InputLayer
from   tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import vision as vision
import math
import numpy as np
import cv2
import time
import json
import random
import argparse
from   collections import OrderedDict

# ------------------------------------------------------------------
# hyper-parameters and mode selection
# ------------------------------------------------------------------
parser        = argparse.ArgumentParser(description="Annotate images with click-based labels.")
parser.add_argument("-s", "--session", required=True,   help="session data (name of session; as named in cwd/sessions dir)")
parser.add_argument("-m", "--mode",    default='track', help='model-type to create; types = target (to obtain eye-center) and track (to infer screen pos, with eye center as paramter in input model)')
parser.add_argument("-q", "--quality", default=2)
parser.add_argument("-r", "--repeat",  default=1)
parser.add_argument("-e", "--epochs",  default=500)
parser.add_argument("-i", "--input",   default=64)
parser.add_argument("-b", "--batch",   default=1)
parser.add_argument("-lr", "--learning-rate", default=0.0001)
args          = parser.parse_args()
session_ids   = args.session.split(',')
quality       = int(args.quality) # top level scaling factor
resize        = int(args.input)   # track: crop (128x128) or entire image (340x340) is resized to 64x64
input_size    = resize
size          = 128          # crop size for track mode
full_size     = 340          # original full image size
batch_size    = int(args.batch)
learning_rate = float(args.learning_rate)
num_epochs    = int(args.epochs)
mode          = args.mode
is_target     = mode == 'target'

class target(vision.model):
    def __init__(self, data, batch_size=1):
        self.data          = data
        self.batch_size    = batch_size

    def label(self, data, i):
        image_path = data.image[i]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        # Get target coordinates
        iris_mid = data.iris_mid[i].copy()
        # Apply augmentations for training
        if data.train:
            # Random horizontal flip
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                iris_mid[0] = -iris_mid[0]  # Flip x-coordinate
            # Random crop & scale
            if random.random() < 0.5:
                new_h = new_w = int(full_size * random.uniform(0.85, 1.0))
                max_trans_x = full_size - new_w
                max_trans_y = full_size - new_h
                start_x = random.randint(0, max_trans_x)
                start_y = random.randint(0, max_trans_y)
                # Apply crop
                image = image[start_y:start_y + new_h, start_x:start_x + new_w]
                # Adjust target coordinates
                tl_x = ((iris_mid[0] + 0.5) * full_size - start_x)
                tl_y = ((iris_mid[1] + 0.5) * full_size - start_y)
                iris_mid = [(tl_x / new_w) - 0.5, (tl_y / new_h) - 0.5]
            
            # Resize and normalize image
            image = cv2.resize(image, (input_size, input_size))
            image = image / 255.0
            image = np.expand_dims(image, axis=-1)
        
        # Return in the format expected by the model
        return {'image': image }, {'iris_mid': np.array(iris_mid)}

    def model(self, quality):
        image = tf.keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = tf.keras.layers.Conv2D       (name='conv0', filters=8 * quality, kernel_size=3, padding='same')(image)
        x = tf.keras.layers.ReLU         (name='relu0')(x)
        x = tf.keras.layers.MaxPooling2D (name='pool0', pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D       (name='conv1', filters=16 * quality, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.ReLU         (name='relu1')(x)
        x = tf.keras.layers.MaxPooling2D (name='pool1', pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten      (name='flatten')(x)
        x = tf.keras.layers.Dense        (name='dense0', units=quality * 64, activation='relu')(x)
        x = tf.keras.layers.Dense        (name='dense1', units=2)(x)
        return tf.keras.Model            (name=mode, inputs=image, outputs=x)


class track(vision.model):
    def __init__(self, data, batch_size=1):
        self.data = data  # "track" or "target"

    def label(self, data, index):
        f2_image_path = data.f2_images[index]
        f6_image_path = data.f6_images[index]
        f2_iris_mid   = data.f2_iris_mid[index]
        f6_iris_mid   = data.f6_iris_mid[index]
        f2_iris_mid   = np.array(f2_iris_mid).astype(np.float32)
        f6_iris_mid   = np.array(f6_iris_mid).astype(np.float32)

        def process_image(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = vision.center_crop(image, size, f2_iris_mid)
            image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
            image = (image / 255.0).astype(np.float32)
            image = np.expand_dims(image, axis=-1)
            return image
        
        f2_image = process_image(f2_image_path)
        f6_image = process_image(f6_image_path)
        pixel    = data.pixel[index].copy()
        return {'f2_image':f2_image, 'f2_iris_mid':f2_iris_mid, 'f6_image':f6_image, 'f6_iris_mid':f6_iris_mid}, pixel
    
    def model(self, quality):
        input_f2          = tf.keras.layers.Input(name='input_f2',          shape=(input_size, input_size, 1))
        input_f2_iris_mid = tf.keras.layers.Input(name='input_f2_iris_mid', shape=(2,))
        input_f6          = tf.keras.layers.Input(name='input_f6',          shape=(input_size, input_size, 1))
        input_f6_iris_mid = tf.keras.layers.Input(name='input_f6_iris_mid', shape=(2,))

        def conv2D_for(id, image, iris_mid):
            x = tf.keras.layers.Conv2D       (name=f'f{id}_conv0', activation='relu', filters=16 * quality, kernel_size=3, padding='same')(image)
            x = tf.keras.layers.MaxPooling2D (name=f'f{id}_pool0', pool_size=(2, 2))(x)
            x = tf.keras.layers.Conv2D       (name=f'f{id}_conv1', activation='relu', filters=32 * quality, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.MaxPooling2D (name=f'f{id}_pool1', pool_size=(2, 2))(x)
            x = tf.keras.layers.Flatten      (name=f'f{id}_flatten')(x)
            return x

        x0 = conv2D_for(2, input_f2, input_f2_iris_mid)
        x1 = conv2D_for(6, input_f6, input_f6_iris_mid)
        x0 = tf.keras.layers.Concatenate (name='concat0')([x0, input_f2_iris_mid]) 
        e0 = tf.keras.layers.Dense       (quality * 32, activation='relu', name="e0")(x0)
        x1 = tf.keras.layers.Concatenate (name='concat1')([x1, input_f6_iris_mid]) 
        e1 = tf.keras.layers.Dense       (quality * 32, activation='relu', name="e1")(x1)

        x  = tf.keras.layers.Concatenate (name='concat2')([e0, e1]) # i can train with x0 or x1, but if i concat them, its worse than combining by a LOT.. makes no sense?
        x  = tf.keras.layers.Dense       (name='dense0', units=quality * 8, activation='relu')(x)
        x  = tf.keras.layers.Dense       (name='dense1', units=2)(x)
        return tf.keras.Model            (name=mode, inputs=[input_f2, input_f2_iris_mid, input_f6, input_f6_iris_mid], outputs=x)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
repeat          = int(args.repeat)
gen             = target if is_target else track
data            = vision.data(session_ids,  validation_split=0.1, repeat=repeat)
train_generator = gen(data=data.train,      batch_size=batch_size)
val_generator   = gen(data=data.validation, batch_size=batch_size)

vision.train(train_generator, val_generator, quality, learning_rate, num_epochs, input_size=input_size)
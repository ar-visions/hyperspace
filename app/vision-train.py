#!/bin/env python3
import tensorflow as tf
from   tensorflow.keras              import KerasTensor
from   tensorflow.keras              import layers
from   tensorflow.keras.layers       import InputLayer
from   tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
parser.add_argument("-f", "--refine",  action="store_true")
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
is_target     = mode == 'target' # target models have zones of scale (with variance) with translation variance
refine        = args.refine      # refine is our 2nd zone


# target model gives us a rough estimate; can be up to 15% error
class target(vision.model):
    def __init__(self, data, batch_size):
        self.mode          = 'target' if not refine else 'refine'
        self.data          = data
        self.batch_size    = batch_size

    # make sure teh annotator can help us fill in annotations for images that have annotations on other camera ids.
    # thats a simple text replacement and check filter

    def label(self, data, i):
        # this model is 1 image at a time, but our batching is setup for the model of all camera annotations
        # so its easiest to set a repeat count of > 4 and perform a random selection
        image_path = data.f2_image     [i]
        eye_left   = data.f2_eye_left  [i]
        eye_right  = data.f2_eye_right [i]
        iris_mid   = data.f2_iris_mid  [i].copy()

        if random.random() < 0.5:
            image_path = data.f6_image     [i]
            eye_left   = data.f6_eye_left  [i]
            eye_right  = data.f6_eye_right [i]
            iris_mid   = data.f6_iris_mid  [i].copy()
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        #if self.mode == 'refine':
        #    new_h   = new_w = int(full_size * random.uniform(0.3, 0.6))
        #    half    = full_size // 2
        #    iris_mid_x = half + int(iris_mid[0] * half)
        #    iris_mid_y = half + int(iris_mid[1] * half)
        #    offset_x   = full_size * random.uniform(0.85, 1.15)
        #    offset_y   = full_size * random.uniform(0.85, 1.15)
        #    start_x    = (half + (iris_mid_x * half) + offset_x)
        #    start_y    = (half + (iris_mid_y * half) + offset_y)
        #    iris_new_x = (iris_mid_x - start_x) / new_w 
        #    iris_new_y = (iris_mid_y - start_y) / new_h
        #    iris_mid   = np.array([iris_new_x - 0.5, iris_new_y - 0.5], dtype=np.float32)
        #    image      = vision.center_crop(image, new_w, iris_mid)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            iris_mid[0] = -iris_mid[0]  # Flip x-coordinate

        image = cv2.resize(image, (input_size, input_size))
        image = image / 255.0
        """
        # output the training data
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        # Convert iris_mid coordinates (-0.5 to 0.5 range) to image pixel coordinates
        h, w = image.shape
        pixel_x = int((iris_mid[0] + 0.5) * w)
        pixel_y = int((iris_mid[1] + 0.5) * h)
        plt.scatter(pixel_x, pixel_y, c='cyan', s=100, marker='x')
        plt.title(f"Iris Midpoint: ({iris_mid[0]:.2f}, {iris_mid[1]:.2f})")
        plt.axhline(y=pixel_y, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=pixel_x, color='r', linestyle='--', alpha=0.5)
        plt.show(block=True)
        """
        image = np.expand_dims(image, axis=-1)
        return {'image': image }, iris_mid

    def loss(self, y_true, y_pred):
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


    def model(self, quality):
        image = tf.keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = tf.keras.layers.Conv2D       (name='conv0', filters=16 * quality, kernel_size=3, padding='same')(image)
        x = tf.keras.layers.ReLU         (name='relu0')(x)
        x = tf.keras.layers.MaxPooling2D (name='pool0', pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D       (name='conv1', filters=32 * quality, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.ReLU         (name='relu1')(x)
        x = tf.keras.layers.MaxPooling2D (name='pool1', pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten      (name='flatten')(x)
        x = tf.keras.layers.Dense        (name='dense0', units=quality * 64, activation='relu')(x)
        x = tf.keras.layers.Dense        (name='dense1', units=2)(x)
        return tf.keras.Model            (name=mode, inputs=image, outputs=x)


    def model2(self, quality):
        image = tf.keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = tf.keras.layers.Conv2D   (name='conv0', filters=8  * quality, strides=1, activation='relu', kernel_size=3, padding='same')(image)
        x = tf.keras.layers.Conv2D   (name='conv1', filters=16 * quality, strides=1, activation='relu', kernel_size=3, padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg')(x)
        x = tf.keras.layers.Dense    (name='dense0', units=quality * 32, activation='relu')(x)
        x = tf.keras.layers.Dense    (name='dense1', units=2,            activation='linear')(x)
        return tf.keras.Model        (name=mode, inputs=image, outputs=x)


# track model has a small image 32x32 covering entire area of face cropped at iris-mid (augmented), then 32x32 for left, and 32x32 for right
# critical to this is a iris-mid xy, which is a crop location in the image.  its a landmark for where we
# are in spatial view.  i dont believe we also need to give eye-l and r coordinates along with it, however
# that may be an option

class track(vision.model):
    def __init__(self, data, batch_size):
        super().__init__(
            mode        = 'track',
            data        =  data,
            batch_size  =  batch_size)
        self.target_model = tf.keras.models.load_model('vision_target.keras')
        self.target_model.summary()

    def loss(self, y_true, y_pred):
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

    def label(self, data, index):
        # perform all augmentation here
        f2_image_path = data.f2_image[index]
        f6_image_path = data.f6_image[index]
        # our track data in center-close is all here, we have no missing data for l/r/iris-mid
        f2_iris_mid   = data.f2_iris_mid[index].copy() if data.f2_iris_mid[index]   is not None else None
        f6_iris_mid   = data.f6_iris_mid[index].copy() if data.f6_iris_mid[index]   is not None else None
        f2_iris_mid_i = data.f2_iris_mid[index].copy() if data.f2_iris_mid_i[index] is not None else None
        f6_iris_mid_i = data.f6_iris_mid[index].copy() if data.f6_iris_mid_i[index] is not None else None
        f2_iris_mid   = np.array(f2_iris_mid).astype(np.float32) if f2_iris_mid is not None else None
        f6_iris_mid   = np.array(f6_iris_mid).astype(np.float32) if f6_iris_mid is not None else None

        # we do not vary the crop location, but rather vary the iris-mid's location and resulting pixel location
        def process_image(image_path, iris_mid):
            whole = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # infer target if we do not know what it is; this is helpful when we do not have the annotations on center for a video sequence
            if not isinstance(iris_mid, (list, tuple, np.ndarray)):
                whole_input = cv2.resize(whole, (32, 32), interpolation=cv2.INTER_AREA)
                whole_input = (whole_input / 255.0).astype(np.float32)
                whole_input = np.expand_dims(whole_input, axis=-1)
                whole_input = np.expand_dims(whole_input, axis=0)
                iris_mid = self.target_model(whole_input).numpy().squeeze() # if we have both, we could mix or use 100% of the annotation

            # Prepare for visualization
            #"""
            plt.figure(figsize=(8, 8))
            plt.imshow(whole, cmap='gray')
            
            # Convert iris_mid coordinates (-0.5 to 0.5 range) to image pixel coordinates
            h, w = whole.shape
            pixel_x = int((iris_mid[0] + 0.5) * w)
            pixel_y = int((iris_mid[1] + 0.5) * h)
            plt.scatter(pixel_x, pixel_y, c='cyan', s=100, marker='x')
            plt.title(f"Iris Midpoint: ({iris_mid[0]:.2f}, {iris_mid[1]:.2f})")
            plt.axhline(y=pixel_y, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=pixel_x, color='r', linestyle='--', alpha=0.5)
            plt.show(block=True)
            #"""
            image = vision.center_crop(whole, size, iris_mid)
            image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
            image = (image / 255.0).astype(np.float32)
            image = np.expand_dims(image, axis=-1)

            return image, iris_mid
        
        f2_image, f2_iris_mid = process_image(f2_image_path, f2_iris_mid_i if f2_iris_mid_i else f2_iris_mid)
        f6_image, f6_iris_mid = process_image(f6_image_path, f6_iris_mid_i if f6_iris_mid_i else f6_iris_mid)
        data.f2_iris_mid[index] = f2_iris_mid
        data.f6_iris_mid[index] = f6_iris_mid
        pixel    = data.pixel[index].copy()
        return {'f2_image':    f2_image,
                'f6_image':    f6_image,
                'f2_iris_mid': f2_iris_mid,
                'f6_iris_mid': f6_iris_mid}, pixel

    def model(self, quality):
        # Inputs
        f2_image    = tf.keras.layers.Input(name='f2_image',    shape=(input_size, input_size, 1))
        f6_image    = tf.keras.layers.Input(name='f6_image',    shape=(input_size, input_size, 1))
        f2_iris_mid = tf.keras.layers.Input(name='f2_iris_mid', shape=(2,))
        f6_iris_mid = tf.keras.layers.Input(name='f6_iris_mid', shape=(2,))

        # Convolutional Block
        def conv2D_for(id, image):
            x = tf.keras.layers.Conv2D      (name=f'f{id}_conv0', strides=2, activation='relu', filters=8 * quality, kernel_size=3, padding='same')(image)
            x = tf.keras.layers.Conv2D      (name=f'f{id}_conv1', strides=2, activation='relu', filters=16 * quality, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.GlobalAveragePooling2D(name=f'f{id}_gap')(x)
            return x

        # Feature extraction
        x0 = conv2D_for(2, f2_image)
        x1 = conv2D_for(6, f6_image)

        # Incorporate the helper offsets
        x0 = tf.keras.layers.Concatenate(name='concat0')([x0, f2_iris_mid])
        x1 = tf.keras.layers.Concatenate(name='concat1')([x1, f6_iris_mid])

        # Per-feature dense layers
        e0 = tf.keras.layers.Dense(quality * 64, activation='relu', name="e0")(x0)
        e1 = tf.keras.layers.Dense(quality * 64, activation='relu', name="e1")(x1)

        # Feature Fusion (Avoid Direct Concatenation)
        x = tf.keras.layers.Add(name='fused_features')([e0, e1])  # Summing is often more stable than concatenation

        # Final MLP
        x = tf.keras.layers.Dense(name='dense0', units=quality * 16, activation='relu')(x)
        x = tf.keras.layers.Dense(name='output', units=2, activation='linear')(x)

        return tf.keras.Model(name="fusion_model", inputs=[f2_image, f6_image, f2_iris_mid, f6_iris_mid], outputs=x)



# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
repeat  = int(args.repeat)
gen     = target if is_target else track
data    = vision.data(mode, session_ids,  validation_split=0.1, repeat=repeat)
train   = gen(data=data.train,      batch_size=batch_size)
val     = gen(data=data.validation, batch_size=batch_size)

vision.train(train, val, quality, learning_rate, num_epochs, input_size=input_size)
#!/bin/env python3
import tensorflow as tf
import tensorflow.keras as keras
from   keras              import KerasTensor
from   keras              import layers
from   keras.layers       import InputLayer
from   keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import vision as vision
import math
import copy
import numpy as np
import cv2
import time
import json
import random
import argparse
from   collections import OrderedDict
from   keras.constraints import Constraint
import keras.backend as K
# ------------------------------------------------------------------
# hyper-parameters and mode selection
# ------------------------------------------------------------------
parser        = argparse.ArgumentParser(description="Annotate images with click-based labels.")
parser.add_argument("-s", "--session", required=True,   help="session data (name of session; as named in cwd/sessions dir)")
parser.add_argument("-m", "--mode",    default='track', help='model-type to create; types = target (to obtain eye-center) and track (to infer screen pos, with eye center as paramter in input model)')
parser.add_argument("-r", "--repeat",  default=1)
parser.add_argument("-e", "--epochs",  default=500)
parser.add_argument("-i", "--input",   default=64)
parser.add_argument("-b", "--batch",   default=1)
parser.add_argument("-lr", "--learning-rate", default=0.0001)
args          = parser.parse_args()
session_ids   = args.session.split(',')
resize        = int(args.input)   # track: crop (128x128) or entire image (340x340) is resized to 64x64
input_size    = resize
size          = 128          # crop size for track mode
full_size     = 340          # original full image size
batch_size    = int(args.batch)
learning_rate = float(args.learning_rate)
num_epochs    = int(args.epochs)
mode          = args.mode

def required_fields(mode):
    if mode == 'base':    return ['iris_mid', 'scale']
    if mode == 'target':  return ['iris_mid', 'scale']
    if mode == 'eyes':    return ['iris_mid', 'eye_left', 'eye_right']
    if mode == 'track':   return ['iris_mid', 'eye_left', 'eye_right']
    if mode == 'refine':  return ['iris_mid']
    return ['iris_mid']

# scale is tedious to plot, so we train based on a variance of images to get this number
# this scale is fed into target training and should likely be used in track as well
class base(vision.model):
    def __init__(self, data, batch_size):
        self.mode          = 'base'
        self.data          = data
        self.batch_size    = batch_size

    def target_vary(self, image, targets, offset, scale_min, scale_max, can_mirror):
        t0 = targets[0]
        iris_mid_x = int((t0[0] + 0.5) * full_size)
        iris_mid_y = int((t0[1] + 0.5) * full_size)

        # Randomly determine crop size
        sc = random.uniform(scale_min, scale_max)
        new_h = new_w = int(full_size * sc)

        # Center the crop properly
        start_x = int(iris_mid_x - new_w / 2 + round(new_w * random.uniform(-offset, offset)))
        start_y = int(iris_mid_y - new_h / 2 + round(new_h * random.uniform(-offset, offset)))

        # Compute new iris-mid in cropped coordinates
        trans = []
        for target in targets:
            x = int((target[0] + 0.5) * full_size)
            y = int((target[1] + 0.5) * full_size)
            translated_x = (x - start_x) / new_w - 0.5
            translated_y = (y - start_y) / new_h - 0.5
            trans.append([translated_x, translated_y])

        # Compute proper crop center
        crop_center = [(start_x + new_w / 2) / full_size - 0.5, (start_y + new_h / 2) / full_size - 0.5]
        image = vision.center_crop(image, new_w, crop_center)
        mirror = False

        # Apply flipping with correct coordinate modification
        if can_mirror and random.random() < 0.5:
            mirror = True
            image = cv2.flip(image, 1)
            for t in trans:
                t[0] = -t[0]  # Flip x-coordinate

        return image, trans, sc, mirror


    # make sure teh annotator can help us fill in annotations for images that have annotations on other camera ids.
    # thats a simple text replacement and check filter
    def label(self, data, i):
        # this model is 1 image at a time, but our batching is setup for the model of all camera annotations
        # so its easiest to set a repeat count of > 4 and perform a random selection
        image_path =  data.f2_image     [i]
        iris_mid   =  data.f2_iris_mid  [i].copy()
        scale      = [data.f2_scale     [i].copy()[0]]

        if random.random() < 0.5:
            image_path = data.f6_image     [i]
            iris_mid   = data.f6_iris_mid  [i].copy()
            scale      = [data.f6_scale     [i].copy()[0]]
        
        image   = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        targets = [iris_mid]
        #image2  = image.copy()
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            for t in targets:
                t[0] = -t[0]  # Flip x-coordinate
        image   = cv2.resize(image, (input_size, input_size))
        image   = image / 255.0
        """
        # output the training data
        plt.figure(figsize=(8, 8))
        plt.imshow(image2, cmap='gray')
        # Convert iris_mid coordinates (-0.5 to 0.5 range) to image pixel coordinates
        h, w = image2.shape
        pixel_x = int((targets[0][0] + 0.5) * w)
        pixel_y = int((targets[0][1] + 0.5) * h)
        plt.scatter(pixel_x, pixel_y, c='cyan', s=100, marker='x')
        plt.title(f"Iris Midpoint: ({targets[0][0]:.2f}, {targets[0][1]:.2f})")
        plt.axhline(y=pixel_y, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=pixel_x, color='r', linestyle='--', alpha=0.5)
        # Draw a circle with radius scale[0] / 2
        radius = (scale[0] / 2) * w  # Convert normalized scale to pixels
        circle = plt.Circle((pixel_x, pixel_y), radius, color='cyan', fill=False, linewidth=2)
        # Add circle to plot
        plt.gca().add_patch(circle)
        plt.show(block=True)
        """

        image = np.expand_dims(image, axis=-1)
        return {'image': image }, [targets[0][0], targets[0][1], scale[0]]
    
    def model(self):
        image = keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = keras.layers.Conv2D       (name='conv0', filters=32, activation='relu', strides=(1,1), kernel_size=3, padding="same")(image)
        x = keras.layers.MaxPooling2D (name='pool0', pool_size=(2,2))(x)
        x = keras.layers.Conv2D       (name='conv1', filters=64, activation='relu', strides=(1,1), kernel_size=3, padding="same")(x)
        x = keras.layers.MaxPooling2D (name='pool1', pool_size=(2,2))(x)
        x = keras.layers.Flatten      (name='flatten')(x)
        x = keras.layers.Dense        (name='dense0', units=8, activation='relu')(x)
        x = keras.layers.Dense        (name='dense1', units=3, activation='tanh')(x)
        return keras.Model            (name=mode, inputs=image, outputs=x)

# annotator could let the user use hte wsad keys and space, to move eye positions to correct spots
# this way we could pre-annotate the image and the user may adjust the plots
# this is an obvious feature to add for quicker, more exact annotations

# we want to target where we get inference from with base
class target(base):
    def __init__(self, data, batch_size):
        self.mode          = 'target'
        self.data          = data
        self.batch_size    = batch_size
        print('target: loading base model')
        self.base_model = keras.models.load_model('vision_base.keras')
        self.base_model.summary()

    # make sure teh annotator can help us fill in annotations for images that have annotations on other camera ids.
    # thats a simple text replacement and check filter
    def label(self, data, i):
        # this model is 1 image at a time, but our batching is setup for the model of all camera annotations
        # so its easiest to set a repeat count of > 4 and perform a random selection
        image_path = data.f2_image     [i]
        eye_left   = data.f2_eye_left  [i]
        eye_right  = data.f2_eye_right [i]
        iris_mid   = data.f2_iris_mid  [i].copy()
        scale      = data.f2_scale     [i].copy()

        if random.random() < 0.5:
            image_path = data.f6_image     [i]
            eye_left   = data.f6_eye_left  [i]
            eye_right  = data.f6_eye_right [i]
            iris_mid   = data.f6_iris_mid  [i].copy()
            scale      = data.f6_scale     [i].copy()
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Get model input shape (assumes model takes single-channel grayscale)
        input_shape = self.base_model.input_shape  # (None, height, width, channels)
        input_height, input_width = input_shape[1:3]

        # Resize cropped region back to model input size
        final_input = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)

        # Normalize (convert to float32 and scale between 0 and 1)
        final_input = final_input.astype(np.float32) / 255.0
        final_input = np.expand_dims(final_input, axis=[0,-1])  # Add batch & channel dim

        # run inference on base model
        predictions = self.base_model(final_input, training=False)
        predictions = tf.convert_to_tensor(predictions).numpy()
        
        # extract iris_mid (x, y) and scale from predictions
        pred_iris_x, pred_iris_y, pred_scale = predictions[0]
        #pred_scale = 0.5
        scale       = [pred_scale]
        scale_min   = (scale[0] * 0.8)
        scale_max   = (scale[0] * 1.2)

        # we may filter in some way if the iris_mid distance is too muc
        # probably better to stage this outside of batching, too.
        image, targets, sc, mirror = self.target_vary(image, [iris_mid], 0.22, scale_min, scale_max, True)
        image2 = image.copy()
        #targets = [iris_mid]
        #new_w = full_size
        image    = cv2.resize(image, (input_size, input_size))
        image    = image / 255.0
        scale[0] = scale[0] / sc

        """
        # output the training data
        plt.figure(figsize=(8, 8))
        plt.imshow(image2, cmap='gray')
        # Convert iris_mid coordinates (-0.5 to 0.5 range) to image pixel coordinates
        h, w = image2.shape
        pixel_x = int((targets[0][0] + 0.5) * w)
        pixel_y = int((targets[0][1] + 0.5) * h)
        plt.scatter(pixel_x, pixel_y, c='cyan', s=100, marker='x')
        plt.title(f"Iris Midpoint: ({targets[0][0]:.2f}, {targets[0][1]:.2f})")
        plt.axhline(y=pixel_y, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=pixel_x, color='r', linestyle='--', alpha=0.5)
        # Draw a circle with radius scale[0] / 2
        radius = (scale[0] / 2) * w  # Convert normalized scale to pixels
        circle = plt.Circle((pixel_x, pixel_y), radius, color='cyan', fill=False, linewidth=2)
        # Add circle to plot
        plt.gca().add_patch(circle)
        plt.show(block=True)
        """
        image    = np.expand_dims(image, axis=-1)
        return {'image': image }, np.array([targets[0][0], targets[0][1], scale[0]])

    def model(self):
        image = keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = keras.layers.Conv2D       (name='conv0', filters=32, activation='relu', strides=(1,1), kernel_size=5, padding="same")(image)
        x = keras.layers.MaxPooling2D (name='pool0', pool_size=(2,2))(x)
        x = keras.layers.Flatten      (name='flatten')(x)
        x = keras.layers.Dense        (name='dense0', units=8, activation='relu')(x)
        x = keras.layers.Dense        (name='dense1', units=3, activation='tanh')(x)
        return keras.Model            (name=mode, inputs=image, outputs=x)

# we want to target where we get inference from with base
class eyes(target):
    def __init__(self, data, batch_size):
        self.mode          = 'target'
        self.data          = data
        self.batch_size    = batch_size
        self.base_model = keras.models.load_model('vision_base.keras')
        self.base_model.summary()
        self.target_model = keras.models.load_model('vision_target.keras')
        self.target_model.summary()


    # make sure teh annotator can help us fill in annotations for images that have annotations on other camera ids.
    # thats a simple text replacement and check filter
    def label(self, data, i):
        # this model is 1 image at a time, but our batching is setup for the model of all camera annotations
        # so its easiest to set a repeat count of > 4 and perform a random selection
        image_path = data.f2_image     [i]
        eye_left   = data.f2_eye_left  [i]
        eye_right  = data.f2_eye_right [i]
        iris_mid   = data.f2_iris_mid  [i].copy()
        scale      = data.f2_scale     [i].copy() if data.f2_scale[i] is not None else None

        if random.random() < 0.5:
            image_path = data.f6_image     [i]
            eye_left   = data.f6_eye_left  [i]
            eye_right  = data.f6_eye_right [i]
            iris_mid   = data.f6_iris_mid  [i].copy()
            scale      = data.f6_scale     [i].copy() if data.f6_scale[i] is not None else None
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Get model input shape (assumes model takes single-channel grayscale)
        input_shape = self.base_model.input_shape  # (None, height, width, channels)
        input_height, input_width = input_shape[1:3]
        final_input = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
        final_input = final_input / 255.0
        final_input = np.expand_dims(final_input, axis=[0,-1])  # Add batch & channel dim
        if scale is None:
            predictions = self.base_model(final_input, training=False)
            pred_iris_x, pred_iris_y, pred_scale = predictions[0]
            scale       = [pred_scale]
        vary        = 0.10
        # accuracy of scale should directly impact accuracy of left and right eyes
        scale_min   = (scale[0] * (1.0 - vary))
        scale_max   = (scale[0] * (1.0 + vary)) # we need to run against a target model, too: lets first test with base and see if it improves with target
        image, targets, sc, mirror = self.target_vary(image, [iris_mid, eye_right, eye_left], 0.10, scale_min, scale_max, True)
        scale[0] = scale[0] / sc
        image       = cv2.resize(image, (input_size, input_size))
        image       = image / 255.0
        image = np.expand_dims(image, axis=-1)
        i1 = 2 if mirror else 1
        i2 = 1 if mirror else 2
        return {'image': image }, np.array([
            targets[0][0], targets[0][1],  targets[i1][0], targets[i1][1],  targets[i2][0], targets[i2][1], scale[0]    
        ])
    
    def model(self):
        image = keras.layers.Input(shape=(input_size, input_size, 1), name='image')
        x = keras.layers.Conv2D       (name='conv0', filters=64, activation='relu', strides=(1,1), kernel_size=5, padding="same")(image)
        x = keras.layers.MaxPooling2D (name='pool0', pool_size=(2, 2))(x)
        x = keras.layers.Flatten      (name='flatten')(x)
        x = keras.layers.Dense        (name='dense0', units=8, activation='relu')(x)
        x = keras.layers.Dense        (name='dense1', units=7, activation='linear')(x)
        return keras.Model            (name=mode, inputs=image, outputs=x)

scales = {}

class track(base):
    def __init__(self, data, batch_size):
        self.mode       = 'base'
        self.data       = data
        self.batch_size = batch_size
        self.base_model = keras.models.load_model('vision_base.keras')
        self.base_model.summary()
        self.target_model = keras.models.load_model('vision_target.keras')
        self.target_model.summary()

    def label(self, data, index):
        # perform all augmentation here
        f2_image_path = data.f2_image    [index]
        f6_image_path = data.f6_image    [index]
        f2_iris_mid   = data.f2_iris_mid [index].copy()
        f6_iris_mid   = data.f6_iris_mid [index].copy() 
        f2_eye_left   = data.f2_eye_left [index].copy()
        f6_eye_left   = data.f6_eye_left [index].copy() 
        f2_eye_right  = data.f2_eye_right[index].copy()
        f6_eye_right  = data.f6_eye_right[index].copy() 
        f2_scale      = data.f2_scale    [index].copy() if data.f2_scale[index] is not None else [None]
        f6_scale      = data.f6_scale    [index].copy() if data.f6_scale[index] is not None else [None]
        rx            = 0 # random.uniform(-0.05, 0.05)
        ry            = 0 # random.uniform(-0.05, 0.05)

        # we do not vary the crop location, but rather vary the iris-mid's location and resulting pixel location
        def process_image(image_path, scale, f, target):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if scale is None:
                global scales
                if image_path in scales:
                    scale = scales[image_path]
                else:
                    image_copy  = (image.copy() / 255.0).astype(np.float32)
                    image_copy  = cv2.resize(image_copy, (32, 32), interpolation=cv2.INTER_AREA)
                    image_copy  = np.expand_dims(image_copy, axis=[0,-1])
                    predictions = self.base_model(image_copy, training=False)
                    pred_iris_x, pred_iris_y, pred_scale = predictions[0]
                    scale = pred_scale
                    scales[image_path] = scale
            
            vary      = 0.00
            scale_min = (scale / f * (1.0 - vary))
            scale_max = (scale / f * (1.0 + vary))
            t         = copy.deepcopy(target)
            t[0][0]  += rx
            t[0][1]  += ry
            image, targets, sc, mirror = self.target_vary(image, t, 0.0, scale_min, scale_max, False)
            image     = (image / 255.0).astype(np.float32)
            image     = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
            image     = np.expand_dims(image, axis=-1)
            scale     = scale / sc
            return image, targets, scale
        
        # we may mirror, but we would have to do so here at the return
        f2_image, [f2_iris_mid ], f2_image_scale = process_image(f2_image_path, f2_scale[0], 1.0, [f2_iris_mid ])
        f2_left,  [f2_left_pos ], f2_left_scale  = process_image(f2_image_path, f2_scale[0], 4.0, [f2_eye_left ])
        f2_right, [f2_right_pos], f2_right_scale = process_image(f2_image_path, f2_scale[0], 4.0, [f2_eye_right])

        f6_image, [f6_iris_mid ], f6_image_scale = process_image(f6_image_path, f6_scale[0], 1.0, [f6_iris_mid ])
        f6_left,  [f6_left_pos ], f6_left_scale  = process_image(f6_image_path, f6_scale[0], 4.0, [f6_eye_left ])
        f6_right, [f6_right_pos], f6_right_scale = process_image(f6_image_path, f6_scale[0], 4.0, [f6_eye_right])
        pixel = data.pixel[index].copy()

        #plt.figure(figsize=(8, 8))
        #plt.imshow(np.squeeze(f2_left),  cmap='gray')
        #plt.imshow(np.squeeze(f2_right), cmap='gray')
        #plt.imshow(np.squeeze(f6_left),  cmap='gray')
        #plt.imshow(np.squeeze(f6_right), cmap='gray')
        #plt.show(block=True)

        def channels(images):
            channels = [np.asarray(img) for img in images]
            return np.concatenate(channels, axis=-1)  # Stack along the channel axis (axis=-1)

        if random.random() > 0.5:
            channels = channels([f2_image, f6_image, f2_left, f6_left, f2_right, f6_right])
            #channels_image = np.expand_dims(np.concatenate([channels[:, :, i] for i in range(channels.shape[-1])], axis=1), axis=-1)  # Shape: (32, 32*6)
            
            #plt.figure(figsize=(8, 8))
            #plt.imshow(channels_image,  cmap='gray')
            #plt.show(block=True)

            return {'channels':  channels,
                    'f2_image_scale': [f2_image_scale],
                    'f6_image_scale': [f6_image_scale],
                    'f2_iris_mid':  f2_iris_mid,
                    'f6_iris_mid':  f6_iris_mid,
                    'f2_left_pos':  f2_left_pos,
                    'f6_left_pos':  f6_left_pos,
                    'f2_right_pos': f2_right_pos,
                    'f6_right_pos': f6_right_pos
            }, [pixel[0] + rx, pixel[1] + ry]
        else:
            def mirror_image(image):  return np.expand_dims(cv2.flip(np.squeeze(image), 1), axis=-1)  # Flip along the vertical axis
            def mirror_position(pos): return [-pos[0], pos[1]]   # Invert X while keeping Y the same
            channels = channels([
                mirror_image(f2_image),
                mirror_image(f6_image),
                mirror_image(f2_right),
                mirror_image(f6_right),
                mirror_image(f2_left),
                mirror_image(f6_left)])
            #channels_image = np.concatenate([channels[:, :, i] for i in range(channels.shape[-1])], axis=1)  # Shape: (32, 32*6)
            #plt.figure(figsize=(8, 8))
            #plt.imshow(channels_image,  cmap='gray')
            #plt.show(block=True)
            return {
                'channels':  channels,
                'f2_image_scale': [f2_image_scale],
                'f6_image_scale': [f6_image_scale],
                'f2_iris_mid':  mirror_position(f2_iris_mid),
                'f6_iris_mid':  mirror_position(f6_iris_mid),
                'f2_left_pos':  mirror_position(f2_right_pos),
                'f6_left_pos':  mirror_position(f6_right_pos),
                'f2_right_pos': mirror_position(f2_left_pos),
                'f6_right_pos': mirror_position(f6_left_pos)
            }, [-pixel[0] + rx, pixel[1] + ry]  # Mirror pixel X value
        
    def model(self):
        channels     = keras.layers.Input(name='channels',     shape=(input_size, input_size, 6))
        f2_image_scale = keras.layers.Input(name='f2_image_scale', shape=(1,))
        f6_image_scale = keras.layers.Input(name='f6_image_scale', shape=(1,))
        f2_iris_mid  = keras.layers.Input(name='f2_iris_mid',  shape=(2,))
        f6_iris_mid  = keras.layers.Input(name='f6_iris_mid',  shape=(2,))
        f2_left_pos  = keras.layers.Input(name='f2_left_pos',  shape=(2,))
        f6_left_pos  = keras.layers.Input(name='f6_left_pos',  shape=(2,))
        f2_right_pos = keras.layers.Input(name='f2_right_pos', shape=(2,))
        f6_right_pos = keras.layers.Input(name='f6_right_pos', shape=(2,))

        #x = keras.layers.Conv2D       (name='conv0', filters=32, activation='relu', strides=(1,1), kernel_size=3, padding="same")(channels)
        x = keras.layers.DepthwiseConv2D(name='conv0', depth_multiplier=64, activation='relu', strides=(1,1), kernel_size=5, padding="same")(channels)
        x = keras.layers.MaxPooling2D (name='pool1', pool_size=(2,2))(x)
        x = keras.layers.Flatten      (name='flatten')(x)
        x = keras.layers.Concatenate  (name='concat', axis=-1)([x, f2_image_scale, f6_image_scale, f2_iris_mid, f6_iris_mid, f2_left_pos, f6_left_pos, f2_right_pos, f6_right_pos])
        x = keras.layers.Dense        (name='dense0', units=64, activation='relu')(x)
        x = keras.layers.Dense        (name='dense1', units=16, activation='relu')(x)
        x = keras.layers.Dense        (name='dense2', units=2, activation='tanh')(x)

        return keras.Model(name="track", inputs=[
            channels, f2_image_scale, f6_image_scale, f2_iris_mid, f6_iris_mid, f2_left_pos, f6_left_pos, f2_right_pos, f6_right_pos
        ], outputs=x)



# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
repeat  = int(args.repeat)
gen     = globals().get(mode)
assert gen != None, "unresolved model"
data    = vision.data(mode, session_ids, required_fields(mode), validation_split=0.1, final_magnitude=2, repeat=repeat)
train   = gen(data=data.train,      batch_size=batch_size)
final   = gen(data=data.final,      batch_size=batch_size)
val     = gen(data=data.validation, batch_size=batch_size)

vision.train(train, val, final, learning_rate, num_epochs)
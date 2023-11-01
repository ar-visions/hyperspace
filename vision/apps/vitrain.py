import json
import os
import numpy             as np
import tensorflow        as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers, initializers, Sequential
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
lr = 0.001

gen_path = './gen'

fields  = ['x','y','z','rx','ry','rz','fov']
labels  = []
sources = []
i=0

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
    image = tf.image.resize(image, [H, W])
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
batch_size = 32
train_data = train_data.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data  = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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


class PosEmbedding(layers.Layer):
    def __init__(self, patch_size, emb_dim):
        super().__init__()
        self.patcher = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.init = initializers.RandomNormal()

    def build(self, inp_dim):
        num_patches = (inp_dim[1]//self.patch_size) * (inp_dim[2]//self.patch_size)
        dim = self.patch_size * self.patch_size * inp_dim[-1] # p1 * p2 * c
        self.dense = EinMix('b np d -> b np e', weight_shape='d e', d=dim, e=self.emb_dim)
        self.pos_embedding = tf.Variable(self.init((1, num_patches+1, self.emb_dim)), trainable=True)
        self.cls_token = tf.Variable(self.init((1, 1, self.emb_dim)), trainable=True)

    def call(self, x, training=False):
        cls_token = tf.repeat(self.cls_token, tf.shape(x)[0], 0)
        x = self.dense(self.patcher(x), training=training)
        x = tf.concat([cls_token, x], axis=1)
        return x + self.pos_embedding

class MHA(Layer):
    def __init__(self, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = layers.Dense(units=inner_dim * 3, use_bias=False)

    def call(self, x, training=False):
        x = self.to_qkv(x, training=training)
        qkv = tf.split(x, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qk = tf.nn.softmax(einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale)
        attn = einsum(qk, v, 'b h i j, b h j d -> b h i d')
        return rearrange(attn, 'b h i d -> b i (h d)')

class TransformerBlock(layers.Layer):
    def __init__(self, heads, dim_heads):
        super().__init__()
        self.mha = MHA(heads, dim_heads)
        self.fc = Sequential([layers.Dense(dim_heads*heads, activation=tf.keras.activations.gelu),
            layers.Dense(dim_heads*heads)])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.mha(self.ln1(x, training=training))
        h = self.ln2(x, training=training)
        return self.fc(h) + x
    
class TransformerEncoder(tf.keras.Model):
    def __init__(self, heads, dim_heads, n):
        super().__init__()
        self.module = [TransformerBlock(heads, dim_heads) for _ in range(n)]

    def call(self, x, training=False):
        for layer in self.module:
            x = layer(x, training=training)
        return x

class ViT(tf.keras.Model):
    def __init__(self, classes, patch, dim, heads, dim_heads, n):
        super().__init__()
        self.embedding = PosEmbedding(patch, dim)
        self.transformer = TransformerEncoder(heads, dim_heads, n)
        self.fc = layers.Dense(classes)

    def call(self, x, training=False):
        x = self.embedding(x, training=training)
        x = self.transformer(x, training=training)
        x = self.fc(x, training=training)
        return x[:, 0, :]


epochs = 10
model = ViT(num_outputs, 8, 64, 4, 32, 3) # classes

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=custom_metrics)

# Train the model
history = model.fit(train_data, epochs=epochs, validation_data=test_data)

# Evaluate the model on test data
loss, mae = model.evaluate(test_data)

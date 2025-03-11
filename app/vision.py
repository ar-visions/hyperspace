import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import KerasTensor
from    tensorflow.keras.layers import InputLayer
from    tensorflow.keras.regularizers import l2
import  matplotlib.pyplot as plt
import  os
import  math
import  numpy as np
import  time
from    collections import OrderedDict
import  cv2
import  json
import  random
from    typing import Tuple, List, Any, Union

# ------------------------------------------------------------------
# color formatting
# ------------------------------------------------------------------
WHITE  = '\033[97m'
GRAY   = '\033[90m'
BLUE   = '\033[94m'
YELLOW = '\033[93m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
PURPLE = '\033[95m'
RESET  = '\033[0m'



SEED = 242  # You can change this to any fixed number

# Seed Python's built-in random module
random.seed(SEED)

# Seed NumPy
np.random.seed(SEED)

# Seed TensorFlow
tf.random.set_seed(SEED)

# Ensure deterministic behavior for cuDNN (may slow down training slightly)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensures single-GPU determinism (optional)




# keras-interfacing model, with batch using our data
class model(keras.utils.Sequence):
    def __init__(self, mode, data, batch_size):
        self.mode         = mode
        self.data         = data
        self.batch_size   = batch_size
        self.inputs       = []

    def loss(self, y_true, y_pred):
        return keras.losses.MeanSquaredError()(y_true, y_pred)

    # process batch
    def __getitem__(self, batch_index):
        start         = batch_index * self.batch_size
        end           = min((batch_index + 1) * self.batch_size, len(self.data))
        batch_data    = {}
        batch_label   = []
        for i in range(start, end):
            data, label = self.label(self.data, i)
            for f in self.inputs:
                if f not in batch_data:
                    batch_data[f] = []
                batch_data[f].append(np.array(data[f], dtype=np.float32))
            batch_label.append(np.array(label, dtype=np.float32))
        for f in batch_data:  batch_data [f] = np.array(batch_data [f], dtype=np.float32)
        return batch_data, np.array(batch_label)
    
    # length of batched data
    def __len__(self):
        if len(self.data) == 0: return 0
        return max(1, (len(self.data) + self.batch_size - 1) // self.batch_size)
    
    # user implements
    def model(self):
        assert False, 'user-implement'

# export based on quantized state
def export_part(mode, layer, part, index):
    w = layer.get_weights()
    weights_id = f"{mode}_{layer.name}_{part}"
    # check for quantized weights (int8)
    if hasattr(layer, 'quantize_config'):
        # Extract scale and zero-point from quantized layers
        if hasattr(layer, 'get_quantize_params'):
            scale = layer.get_quantize_params().get("scale", 1.0)
            zero_point = layer.get_quantize_params().get("zero_point", 0)
        else:
            scale = 1.0
            zero_point = 0
        # extract scale and zero-point from quantized layers
        #scale      = layer.get_quantize_params().get("scale",      1.0)  # Default to 1.0 if missing
        #zero_point = layer.get_quantize_params().get("zero_point", 0)    # Default to 0
        with open('%s.i8' % weights_id, 'wb') as f:
            np.array(scale, dtype=np.float32).tofile(f)      # Store scale
            np.array(zero_point, dtype=np.float32).tofile(f) # Store zero-point
            w[index].astype(np.int8).tofile(f)               # Store quantized weights
    else:
        # default float weight export
        with open('%s.f32' % weights_id, 'wb') as f:
            w[index].tofile(f)

    return weights_id

def export_json(model, mode, output_json_path, input_shapes):
    #return
    layers_dict = OrderedDict()
    output_shapes = {}
    # Function to get layer type
    def get_layer_type(layer):
        layer_class = layer.__class__.__name__
        if   layer_class == 'Conv2D':       return "conv"
        elif layer_class == 'ReLU':         return "relu"
        elif layer_class == 'MaxPooling2D': return "pool"
        elif layer_class == 'Dense':        return "dense"
        elif layer_class == 'Flatten':      return "flatten"
        elif layer_class == 'InputLayer':   return "input"
        elif layer_class == 'Concatenate':  return "concatenate"
        else:
            return layer_class
    
    # Run a sample inference to ensure all shapes are computed
    #if mode == 'target':
    #    sample_input  = np.zeros((1, input_size, input_size, 1))
    #    _ = model(sample_input)
    #else:
    #    sample_image  = np.zeros((1, input_size, input_size, 1))
    #    sample_iris_mid = np.zeros((1, 2))
    #    _ = model([sample_image, sample_iris_mid])
    
    def input_shape(layer):
        if hasattr(layer.input, 'shape'):
            if isinstance(layer.input, list):
                return [tensor.shape[1:] for tensor in layer.input]
            return layer.input.shape[1:]
        return None
    
    def get_output_shape(layer):
        if hasattr(layer.output, 'shape'):
            if isinstance(layer.output, list):
                return [list(tensor.shape[1:]) for tensor in layer.output]
            return list(layer.output.shape[1:])
        return None
    
    # Now the shapes should be populated
    for i, layer in enumerate(model.layers):
        layer_class = layer.__class__.__name__
        ishape      = input_shape(layer)
        oshape      = get_output_shape(layer)
        idim        = layer.input.shape[-1] if hasattr(layer.input, 'shape') and not isinstance(layer.input, list) else None
        
        if oshape is not None:
            output_shapes[layer.name] = oshape
        if hasattr(layer, '_inbound_nodes') and layer._inbound_nodes:
            for node in layer._inbound_nodes:
                if hasattr(node, 'inbound_layers'):
                    print(f"  Inbound layers: {[l.name for l in node.inbound_layers]}")
        
        layers_dict[layer.name] = { "name": layer.name }
        if   layer_class == 'InputLayer':
            layers_dict[layer.name].update({
                "Type":         "input",
                "input":        input_shapes
            })
        elif layer_class == 'Conv2D':
            layers_dict[layer.name].update({
                "Type":         "conv",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "in_channels":  idim,
                "out_channels": layer.filters,
                "kernel_size":  list(layer.kernel_size),
                "strides":      list(layer.strides),
                "padding":      layer.padding,
                "activation":   layer.activation.__name__,
                "weights":      export_part(mode, layer, 'weights', 0),  # Kernel weights
                "bias":         export_part(mode, layer, 'bias',    1) # initial value in the accumulator in gemm
            })
        elif layer_class == 'MaxPooling2D':
            layers_dict[layer.name].update({
                "Type":         "pool",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "type":         "max",
                "pool_size":    list(layer.pool_size),
                "strides":      list(layer.strides)
            })
        elif layer_class == 'Dense':
            layers_dict[layer.name].update({
                "Type":         "dense",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "input_dim":    idim,
                "output_dim":   layer.units,
                "activation":   layer.activation.__name__,
                "weights":      export_part(mode, layer, 'weights', 0),
                "bias":         export_part(mode, layer, 'bias', 1)
            })
        elif layer_class == 'Concatenate':
            layers_dict[layer.name].update({
                "Type":         "concatenate",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "axis":         layer.axis
            })
        elif layer_class == 'ReLU':
            layers_dict[layer.name].update({
                "Type":         "relu",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "threshold":    layer.threshold
            })
        elif layer_class == 'Flatten':
            layers_dict[layer.name].update({
                "Type":         "flatten",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else []
            })
        else:
            print(f'ai: implement layer_class: {layer_class}')
        

        # Dictionary to track which layers output to which others
        output_connections = {layer.name: set() for layer in model.layers}

    # Process layers once to extract inputs
    for layer in model.layers:
        # Gather unique input layers
        input_layers = set()

        # Use model's internal config to extract connections
        for node in layer._inbound_nodes:
            for inbound_layer in getattr(node, 'inbound_layers', []):
                input_layers.add(inbound_layer.name)
                output_connections[inbound_layer.name].add(layer.name)

        # Save the unique inputs to the layer dictionary
        layers_dict[layer.name]["inputs"] = list(input_layers)

    # Identify terminal nodes (layers that do not output to any others)
    remaining_nodes = [layer for layer in layers_dict if not output_connections[layer]]
    
    assert len(remaining_nodes), "endless loop"
    output_tensor = []
    for node in remaining_nodes:
        if node in output_shapes:
            output_tensor.append(output_shapes[node])
    layers_dict["output"] = {
        "name":         "output",
        "Type":         "output",
        "inputs":       remaining_nodes,
        "tensor":       output_tensor
    }
    ops        = list(layers_dict.values())
    model_json = {
        "ident":            mode,
        "output":           [[2]],
        "ops":              ops
    }

    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(model_json, f, indent=4)


def train(train, val, final, lr, epochs):
    model = train.model()
    model.summary()

    # prevent boilerplate with this round-about input binding
    train.inputs = []
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            train.inputs.append(layer.name)
    val.inputs   = train.inputs
    final.inputs = train.inputs
    # Set up optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    bar_length = 40
    input_shape = None
    output_dim = model.output_shape[-1]

    # training and validation loops
    for epoch in range(epochs):
        final_epochs = epoch >= (epochs // 2)
        f_train = final if final_epochs else train
        model.trainable = True
        total_loss = 0.0
        total_errors = tf.zeros(output_dim)  # Accumulate total X, Y errors
        num_batches = len(f_train)
        mode = train.__class__.__name__
        
        #if final_epochs: print('final epochs mode')
        for i in range(num_batches):
            try:
                data, labels = f_train[i]
                inputs   = {}
                for name in f_train.inputs:
                    inputs[name] = data[name]
                if not input_shape:
                    input_shape = [inputs[name].shape[1:] for name in f_train.inputs]
                
                outputs = model(inputs, training=True)

                # forward pass and calculate gradients
                with tf.GradientTape() as tape:
                    outputs = model(inputs, training=True)
                    loss    = f_train.loss(labels, outputs)
                
                # Backward pass and update weights
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Calculate metrics
                total_loss   += loss.numpy()
                batch_error   = tf.reduce_mean(tf.abs(outputs - labels), axis=0)
                total_errors += batch_error
                progress      = (i + 1) / num_batches
                avg_loss      = total_loss / (i + 1)
                avg_errors    = (total_errors / (i + 1)).numpy()
                fin           = progress >= 0.9999
                PCOLOR        = GREEN if fin else CYAN
                final_training = 'finalize' if final_epochs else 'training'
                labels = ['X', 'Y', 'Z', 'W', 'AA', 'BB', 'CC'][:len(avg_errors)]  # Only take as many labels as needed
                error_str = ", ".join(f"{label}={error:.4f}" for label, error in zip(labels, avg_errors))
                status        = (f'\r{YELLOW}{final_training} {GRAY}{BLUE}{epoch+1:4}/{epochs:4}{GRAY} | ' +
                        PCOLOR + '━' * math.floor(progress * bar_length) + f'╸{GRAY}' +
                        '━' * math.ceil((1.0 - progress) * bar_length - 1) +
                        f'{GRAY} | {YELLOW}{avg_loss:.4f} {GRAY}|{BLUE} {error_str}{RESET}')
                
                print(status, end='')
            except Exception as e:
                print(f"\nerror processing batch {i}: {str(e)}")
                continue
        
        # Validation phase
        model.trainable  = False
        val_loss         = 0.0
        total_val_errors = tf.zeros(output_dim)
        num_val_batches  = len(val)
        
        # Iterate through each validation batch
        for j in range(num_val_batches):
            try:
                # Unpack the validation data
                if mode == 'target':
                    val_images, val_labels = val[j]
                    #val_images = val_images[model_inputs[0]] if len(model_inputs) == 1 else [val_images[name] for name in model_inputs]

                    val_outputs   = model(val_images, training=False)
                else:
                    val_inputs, val_labels = val[j]
                    val_outputs   = model(val_inputs, training=False)
                
                # Calculate validation loss and errors
                val_step_loss     = train.loss(val_labels, val_outputs).numpy()
                val_loss         += val_step_loss
                val_batch_error   = tf.reduce_mean(tf.abs(val_outputs - val_labels), axis=0)
                total_val_errors += val_batch_error
            except Exception as e:
                print(f"\nError processing validation batch {j}: {str(e)}")
                continue
        
        # Calculate final validation metrics
        if num_val_batches > 0:
            avg_val_loss = val_loss / num_val_batches
            avg_val_errors = (total_val_errors / num_val_batches).numpy()
        else:
            avg_val_loss = 0
            avg_val_errors = [0, 0]
        
        # Print validation results on a new line
        labels = ['X', 'Y', 'Z', 'W', 'AA', 'BB', 'CC'][:len(avg_val_errors)]  # Only take as many labels as needed
        error_str = ", ".join(f"{label}={error:.4f}" for label, error in zip(labels, avg_val_errors))
    
        print(f'\n{BLUE}validate' + ' ' * 57 +
              f'{avg_val_loss:.4f} {GRAY}|{PURPLE} {error_str}{RESET}\n')
    
    model.save(f'vision_{mode}.keras')
    export_json(model, mode, f'vision_{mode}.json', input_shape)


def get_annots(session_dir, filename, id):
    image_name = filename.replace('f2_', f'f{id}_')
    image_path = os.path.join(session_dir, image_name)
    json_path  = os.path.join(session_dir, image_name.replace(".png", ".json"))
    iris_mid   = None
    eye_left   = None
    eye_right  = None
    scale      = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
            labels    = json_data['labels']
            iris_mid  = labels.get('iris-mid')
            eye_left  = labels.get("eye-left")
            eye_right = labels.get("eye-right")
            scale     = labels.get("scale")

    return image_path, iris_mid, eye_left, eye_right, scale


def shuffle(
    dataset_obj: Any,
    validation_split: float = 0.1,
    repeat: int = 1
) -> Tuple[Any, Any]:
    # get all fields that are lists, check for same length
    fields = [field for field in dir(dataset_obj) if (not field.startswith('__') and field != 'train') and isinstance(getattr(dataset_obj, field), list)]
    field_lengths = [len(getattr(dataset_obj, field)) for field in fields]
    if len(set(field_lengths)) > 1:
        raise ValueError(f"All data fields must have the same length. Got lengths: {field_lengths}")
    
    # dataset instances
    train_dataset      = dataset_obj.__class__(True)
    validation_dataset = dataset_obj.__class__(False)
    if not fields or not field_lengths[0]:
        return train_dataset, validation_dataset
    
    # Extract data from each field
    lists = [getattr(dataset_obj, field) for field in fields]
    
    # Combine all fields together to maintain alignment
    combined = list(zip(*lists))

    # Shuffle dataset to randomize order
    random.shuffle(combined)

    # Calculate split point
    base_count = len(combined)
    validation_count = round(base_count * validation_split)
    index = base_count - validation_count
    val_combined = combined[index:]

    # **Repeat Entire Combined Dataset Instead of Individual Fields**
    train_combined = combined[:index] * repeat  # Repeat whole entries to maintain alignment
    random.shuffle(train_combined)  # Shuffle again to mix repeated data properly
    
    # Unpack aligned data back into their respective fields
    for i, field in enumerate(fields):
        setattr(train_dataset, field, [row[i] for row in train_combined])
        setattr(validation_dataset, field, [row[i] for row in val_combined])

    return train_dataset, validation_dataset


def center_crop(image, size, iris_mid):
    full_size  = image.shape[0]  # square images
    crop_x     = int((iris_mid[0] + 0.5) * full_size)
    crop_y     = int((iris_mid[1] + 0.5) * full_size)
    start_x    = crop_x - size // 2
    start_y    = crop_y - size // 2
    end_x      = start_x + size
    end_y      = start_y + size
    pad_top    = max(0, -start_y)
    pad_left   = max(0, -start_x)
    pad_bottom = max(0, end_y - full_size)
    pad_right  = max(0, end_x - full_size)
    start_x    = max(0, start_x)
    start_y    = max(0, start_y)
    end_x      = min(full_size, end_x)
    end_y      = min(full_size, end_y)

    # If the entire crop is out of bounds, return a blank image
    if start_x >= full_size or start_y >= full_size or end_x <= 0 or end_y <= 0:
        return np.zeros((size, size), dtype=image.dtype)

    # Extract the valid crop from the image
    crop = image[start_y:end_y, start_x:end_x]

    # If the extracted crop is empty, return a black image
    if crop.size == 0:
        return np.zeros((size, size), dtype=image.dtype)

    # Apply padding if needed
    padded_crop = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0)
    return padded_crop

import copy

class dataset:
    def __init__(self, train):
        self.train         = train
        self.f2_image      = []
        self.f2_iris_mid   = []
        self.f2_eye_left   = []
        self.f2_eye_right  = []
        self.f2_scale      = []
        self.f6_image      = []
        self.f6_iris_mid   = []
        self.f6_eye_left   = []
        self.f6_eye_right  = []
        self.f6_scale      = []
        self.pixel         = []
        self.f2_iris_mid_i = []
        self.f6_iris_mid_i = []
    def __len__(self):
        return len(self.f2_image)
    def copy(self): return copy.deepcopy(self)

    def augment_final(self, magnitude=0):
        """Appends data from train to final when iris_mid[0] > 0.5."""
        independent = self.f2_iris_mid.copy()
        for m in range(magnitude):
            for index, iris_mid in enumerate(independent):  
                if abs(iris_mid[0]) > 0.33 or abs(iris_mid[1]) > 0.33:  # set on model, the caller has access
                    # Append data from self.train to all lists in self.final
                    self.f2_image.append(self.f2_image[index])
                    self.f2_iris_mid.append(self.f2_iris_mid[index])
                    self.f2_eye_left.append(self.f2_eye_left[index])
                    self.f2_eye_right.append(self.f2_eye_right[index])
                    self.f2_scale.append(self.f2_scale[index])
                    self.f6_image.append(self.f6_image[index])
                    self.f6_iris_mid.append(self.f6_iris_mid[index])
                    self.f6_eye_left.append(self.f6_eye_left[index])
                    self.f6_eye_right.append(self.f6_eye_right[index])
                    self.f6_scale.append(self.f6_scale[index])
                    self.pixel.append(self.pixel[index])
                    self.f2_iris_mid_i.append(self.f2_iris_mid_i[index])
                    self.f6_iris_mid_i.append(self.f6_iris_mid_i[index])

    def shuffle_data(self):
        """Shuffles all lists together to maintain data integrity."""
        combined = list(zip(
            self.f2_image, self.f2_iris_mid, self.f2_eye_left, self.f2_eye_right, self.f2_scale,
            self.f6_image, self.f6_iris_mid, self.f6_eye_left, self.f6_eye_right, self.f6_scale,
            self.pixel, self.f2_iris_mid_i, self.f6_iris_mid_i
        ))
        np.random.shuffle(combined)  # Shuffle while keeping lists aligned
        (
            self.f2_image, self.f2_iris_mid, self.f2_eye_left, self.f2_eye_right, self.f2_scale,
            self.f6_image, self.f6_iris_mid, self.f6_eye_left, self.f6_eye_right, self.f6_scale,
            self.pixel, self.f2_iris_mid_i, self.f6_iris_mid_i
        ) = zip(*combined)  # Unpack back to separate lists

        # Convert back to lists after shuffling
        self.f2_image       = list(self.f2_image)
        self.f2_iris_mid    = list(self.f2_iris_mid)
        self.f2_eye_left    = list(self.f2_eye_left)
        self.f2_eye_right   = list(self.f2_eye_right)
        self.f2_scale       = list(self.f2_scale)
        self.f6_image       = list(self.f6_image)
        self.f6_iris_mid    = list(self.f6_iris_mid)
        self.f6_eye_left    = list(self.f6_eye_left)
        self.f6_eye_right   = list(self.f6_eye_right)
        self.f6_scale       = list(self.f6_scale)
        self.pixel          = list(self.pixel)
        self.f2_iris_mid_i  = list(self.f2_iris_mid_i)
        self.f6_iris_mid_i  = list(self.f6_iris_mid_i)

class data:
    def __init__(self, mode, session_ids, required_fields, validation_split=0.1, final_magnitude=1, repeat=1):
        # do not expose all data at once, that would allow train to leak with validation
        _dataset = dataset(False)
        for session_id in session_ids:
            session_dir = f'sessions/{session_id}'
            assert os.path.exists(session_dir), f'dir does not exist: {session_dir}'
            for filename in os.listdir(session_dir):
                if filename.startswith("f2_") and filename.endswith(".png"):
                    parts = filename.replace(".png", "").split("_")
                    pixel_x = pixel_y = 0

                    if len(parts) == 4:
                        _, _, pixel_x, pixel_y = parts
                    
                    elif len(parts) >= 7:
                        _, _, pixel_x, pixel_y, _, _, _ = parts
                    else:
                        continue

                    pixel_x = float(pixel_x) - 0.5
                    pixel_y = float(pixel_y) - 0.5

                    # we need json and image-path for each f2, f6
                    # this model takes in two images, however should support different ones provided, to turn the inputs on and off (omit from model)
                    f2_image_path, f2_iris_mid, f2_eye_left, f2_eye_right, f2_scale = get_annots(session_dir, filename, 2)
                    f6_image_path, f6_iris_mid, f6_eye_left, f6_eye_right, f6_scale = get_annots(session_dir, filename, 6)
                    
                    # based on mode, this will need to be a filter (define on model and give to data?)
                    # then data->model calls the model?
                    # it is more direct
                    #if not f2_iris_mid or not f6_iris_mid or not f2_scale or not f6_scale:
                    #    continue
                    require_both = mode == 'track'
                    use_f2  = (not 'iris_mid'  in required_fields or f2_iris_mid  is not None) and \
                              (not 'eye_left'  in required_fields or f2_eye_left  is not None) and \
                              (not 'eye_right' in required_fields or f2_eye_right is not None) and \
                              (not 'scale'     in required_fields or f2_scale     is not None)

                    use_f6  = (not 'iris_mid'  in required_fields or f6_iris_mid  is not None) and \
                              (not 'eye_left'  in required_fields or f6_eye_left  is not None) and \
                              (not 'eye_right' in required_fields or f6_eye_right is not None) and \
                              (not 'scale'     in required_fields or f6_scale is not None)

                    # some models require both inupts, so we must also support those modes
                    if (require_both and (not use_f2 or not use_f6)) or (not use_f2 and not use_f6):
                        continue
                    
                    # we always make 2 sets of dataset entries, f2 and f6's so they are always lined up. when we dont have the data we must do this
                    if not use_f2:
                        f2_image_path = f6_image_path
                        f2_iris_mid   = f6_iris_mid
                        f2_eye_left   = f6_eye_left
                        f2_eye_right  = f6_eye_right
                        f2_scale      = f6_scale
                        use_f2        = True
                    
                    if use_f2:
                        _dataset.f2_image     .append(f2_image_path)
                        _dataset.f2_iris_mid  .append(f2_iris_mid)
                        _dataset.f2_eye_left  .append(f2_eye_left)
                        _dataset.f2_eye_right .append(f2_eye_right)
                        _dataset.f2_scale     .append(f2_scale)
                        _dataset.f2_iris_mid_i.append(None) # this is so the labeler can perform an inference on this image, then store the result for next use (epoch)
                        if not use_f6:
                            f6_image_path = f2_image_path
                            f6_iris_mid   = f2_iris_mid
                            f6_eye_left   = f2_eye_left
                            f6_eye_right  = f2_eye_right
                            f6_scale      = f2_scale

                    _dataset.f6_image     .append(f6_image_path)
                    _dataset.f6_iris_mid  .append(f6_iris_mid)
                    _dataset.f6_eye_left  .append(f6_eye_left)
                    _dataset.f6_eye_right .append(f6_eye_right)
                    _dataset.f6_scale     .append(f6_scale)
                    _dataset.f6_iris_mid_i.append(None)

                    _dataset.pixel        .append([pixel_x, pixel_y])


        # we want self.train to contain the same type as _dataset, with its arrays occupied.  shuffle will go through the object fields the user put in (making sure we dont also look at python base fields??)
        self.train, self.validation = shuffle(_dataset, repeat=repeat, validation_split=validation_split)
        self.final = self.train.copy()
        self.final.augment_final(final_magnitude)
        self.final.shuffle_data()
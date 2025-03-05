import  tensorflow as tf
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



SEED = 42  # You can change this to any fixed number

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
class model(tf.keras.utils.Sequence):
    def __init__(self, mode, data, batch_size):
        self.mode         = mode
        self.data         = data
        self.batch_size   = batch_size
        self.inputs       = []

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


def export_part(mode, layer, part, index):
    w = layer.get_weights()
    if   w[0].dtype == np.float32: weight_ext = '.f32'
    elif w[0].dtype == np.float16: weight_ext = '.f16'
    elif w[0].dtype == np.uint8:   weight_ext = '.u8'
    weights_filename = f"{mode}_{layer.name}_{part}{weight_ext}"
    with open(weights_filename, 'wb') as f: w[index].tofile(f)
    return weights_filename

def export_json(model, mode, output_json_path, quality, input_size):
    return
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
    if mode == 'target':
        sample_input  = np.zeros((1, input_size, input_size, 1))
        _ = model(sample_input)
    else:
        sample_image  = np.zeros((1, input_size, input_size, 1))
        sample_iris_mid = np.zeros((1, 2))
        _ = model([sample_image, sample_iris_mid])
    
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
            })
        elif layer_class == 'Conv2D':
            layers_dict[layer.name].update({
                "Type":         "conv",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "in_channels":  idim,
                "out_channels": layer.filters,
                "kernel_size":  list(layer.kernel_size),
                "padding":      layer.padding,
                "weights":      export_part(mode, layer, 'weights', 0),  # Kernel weights
                "bias":         export_part(mode, layer, 'bias',    1) # initial value in the accumulator in gemm
            })
        elif layer_class == 'MaxPooling2D':
            layers_dict[layer.name].update({
                "Type":         "pool",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "type":         "max",
                "kernel_size":  list(layer.pool_size),
                "stride":       list(layer.strides)
            })
        elif layer_class == 'Dense':
            layers_dict[layer.name].update({
                "Type":         "dense",
                "inputs":        [],
                "tensor":        [ishape] if ishape is not None else [],
                "input_dim":    idim,
                "output_dim":   layer.units,
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
        
    # Build a separate dictionary to track which layers output to which other layers
    # This won't be exported but is used to find terminal nodes
    output_connections = {layer_name: [] for layer_name in layers_dict.keys()}
    model_config = model.get_config()
    if 'layers' in model_config:
        for layer_config in model_config['layers']:
            layer_name = layer_config.get('name')
            if layer_name not in layers_dict:
                continue
            if 'inbound_nodes' in layer_config and layer_config['inbound_nodes']:
                for node in layer_config['inbound_nodes']:
                    for inbound_info in node:
                        if isinstance(inbound_info, list) and len(inbound_info) > 0:
                            source_layer = inbound_info[0]
                            if source_layer not in layers_dict[layer_name]["inputs"]:
                                layers_dict[layer_name]["inputs"].append(source_layer)
                            if source_layer in output_connections:
                                output_connections[source_layer].append(layer_name)
    
    # If the above doesn't work, try direct inspection
    #if all(len(layer_info["inputs"]) == 0 for layer_info in layers_dict.values() if not 'Type' in layer_info or layer_info["Type"] != "input"):
    for i, layer in enumerate(model.layers):
        if i > 0:  # Skip the first layer (usually input)
            input_tensors = layer.input
            if isinstance(input_tensors, list):
                for input_tensor in input_tensors:
                    for prev_layer in model.layers:
                        if prev_layer.output is input_tensor:
                            layers_dict[layer.name]["inputs"].append(prev_layer.name)
                            output_connections[prev_layer.name].append(layer.name)
            else:
                for prev_layer in model.layers:
                    if prev_layer.output is input_tensors:
                        layers_dict[layer.name]["inputs"].append(prev_layer.name)
                        output_connections[prev_layer.name].append(layer.name)
    
    # If still no connections, use a last resort approach
    #if all(len(layer_info["inputs"]) == 0 for layer_info in layers_dict.values() if layer_info["Type"] != "input"):
    for i in range(1, len(model.layers)):
        current_layer = model.layers[i]
        prev_layer    = model.layers[i - 1]
        layers_dict[current_layer.name]["inputs"].append(prev_layer.name)
        output_connections[prev_layer.name].append(current_layer.name)
    
    # Find terminal nodes (layers that don't output to any other layer)
    remaining_nodes = []
    for layer_name, outputs in output_connections.items():
        if len(outputs) == 0 and layer_name in layers_dict:
            remaining_nodes.append(layer_name)
    
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
        "quality":          quality,
        "input":            [[input_size, input_size, 1]] if mode == 'target' else [[input_size, input_size, 1], [2]],
        "output":           [[2]],
        "ops":              ops
    }

    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(model_json, f, indent=4)


def train(train, val, quality, lr, epochs, input_size):
    model = train.model(quality)

    # prevent boilerplate with this round-about input binding
    train.inputs = []
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            train.inputs.append(layer.name)
    model_inputs = train.inputs
    val.inputs   = train.inputs

    # Set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    bar_length = 40

    # Training and validation loops
    for epoch in range(epochs):
        # Training phase
        model.trainable = True
        total_loss = 0.0
        total_errors = tf.zeros(2)  # Accumulate total X, Y errors
        num_batches = len(train)
        mode = train.__class__.__name__

        for i in range(num_batches):
            # Get the data for this batch
            try:
                data, labels = train[i]
                inputs = {}
                for name in model_inputs:
                    inputs[name] = data[name]
                outputs = model(inputs, training=True)

                # forward pass and calculate gradients
                with tf.GradientTape() as tape:
                    outputs = model(inputs, training=True)
                    loss    = train.loss(labels, outputs)
                
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
                status        = (f'\r{YELLOW}training {GRAY}{BLUE}{epoch+1:4}/{epochs:4}{GRAY} | ' +
                        PCOLOR + '━' * math.floor(progress * bar_length) + f'╸{GRAY}' +
                        '━' * math.ceil((1.0 - progress) * bar_length - 1) +
                        f'{GRAY} | {YELLOW}{avg_loss:.4f} {GRAY}|{BLUE} X={avg_errors[0]:.4f}, Y={avg_errors[1]:.4f}{RESET}')
                
                print(status, end='')
            except Exception as e:
                print(f"\nError processing batch {i}: {str(e)}")
                continue
        
        # Validation phase
        model.trainable = False
        val_loss = 0.0
        total_val_errors = tf.zeros(2)
        num_val_batches = len(val)
        
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
        print(f'\n{BLUE}validate' + ' ' * 57 +
              f'{avg_val_loss:.4f} {GRAY}|{PURPLE} X={avg_val_errors[0]:.4f}, Y={avg_val_errors[1]:.4f}{RESET}\n')
    
    model.save(f'vision_{mode}.keras')
    export_json(model, mode, f'vision_{mode}.json', quality, input_size) # todo: read the inputs from model!
    model.summary()


def get_annots(session_dir, filename, id):
    image_name = filename.replace('f2_', f'f{id}_')
    image_path = os.path.join(session_dir, image_name)
    json_path  = os.path.join(session_dir, image_name.replace(".png", ".json"))
    iris_mid   = None
    eye_left   = None
    eye_right  = None

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
            labels    = json_data['labels']
            iris_mid  = labels.get('iris-mid')
            eye_left  = labels.get("eye-left")
            eye_right = labels.get("eye-right")
    
    return image_path, iris_mid, eye_left, eye_right


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

class dataset:
    def __init__(self, train):
        self.train         = train
        self.f2_image      = []
        self.f2_iris_mid   = []
        self.f2_eye_left   = []
        self.f2_eye_right  = []
        self.f6_image      = []
        self.f6_iris_mid   = []
        self.f6_eye_left   = []
        self.f6_eye_right  = []
        self.pixel         = []
        self.f2_iris_mid_i = []
        self.f6_iris_mid_i = []
    def __len__(self):
        return len(self.f2_image)
    
class data:
    def __init__(self, mode, session_ids, validation_split=0.1, repeat=1):
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
                    f2_image_path, f2_iris_mid, f2_eye_left, f2_eye_right = get_annots(session_dir, filename, 2)
                    f6_image_path, f6_iris_mid, f6_eye_left, f6_eye_right = get_annots(session_dir, filename, 6)
                    
                    # based on mode, this will need to be a filter (define on model and give to data?)
                    # then data->model calls the model?
                    # it is more direct
                    if not f2_iris_mid or not f6_iris_mid:
                        continue

                    _dataset.f2_image     .append(f2_image_path)
                    _dataset.f2_iris_mid  .append(f2_iris_mid)
                    _dataset.f2_eye_left  .append(f2_eye_left)
                    _dataset.f2_eye_right .append(f2_eye_right)
                    _dataset.f6_image     .append(f6_image_path)
                    _dataset.f6_iris_mid  .append(f6_iris_mid)
                    _dataset.f6_eye_left  .append(f6_eye_left)
                    _dataset.f6_eye_right .append(f6_eye_right)
                    _dataset.pixel        .append([pixel_x, pixel_y])
                    _dataset.f2_iris_mid_i.append(None) # this is so the labeler can perform an inference on this image, then store the result for next use (epoch)
                    _dataset.f6_iris_mid_i.append(None)


        # we want self.train to contain the same type as _dataset, with its arrays occupied.  shuffle will go through the object fields the user put in (making sure we dont also look at python base fields??)
        self.train, self.validation = shuffle(_dataset, repeat=repeat, validation_split=validation_split)
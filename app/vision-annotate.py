#!/bin/env python3
import os
import json
import random
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Annotate images with click-based labels.")
parser.add_argument("-s", "--session",   required=True, help="session data (name of session; as named in cwd/sessions dir)")
parser.add_argument("-a", "--attribute", required=True, help="eye-center label (between eyes on nose bridge)")
parser.add_argument("-r", "--review",    action="store_true", help="review and edit annotations already made")

args         = parser.parse_args()
a_name       = args.attribute
session      = args.session
review       = args.review
IMAGE_DIR    = "sessions/" + session
dialog       = {}
start_x      = 0
start_y      = 0
canvas_w     = 0
canvas_h     = 0

def normalize_click(x, y, img_w, img_h):
    return [(x - img_w / 2) / img_w, (y - img_h / 2) / img_h]

def json_path(img_path): return os.path.splitext(img_path)[0] + ".json"

def set_annot(img_path, name, object):
    path = json_path(img_path)
    if os.path.exists(path):
        with open(path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {"labels": []}

    updated = False
    for label in json_data["labels"]:
        if name in label:
            label[name] = object
            updated     = True
            break
    if not updated: json_data["labels"].append({name: object})
    with open(path, "w") as f: json.dump(json_data, f, indent=4)

def get_annot(img_path, name):
    path = json_path(img_path)
    if os.path.exists(path):
        with open(path, "r") as f:
            json_data = json.load(f)
        for label in json_data["labels"]:
            if name in label:
                return label[name]
    return None

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_path, img_w, img_h = param  # Unpack params

        # Convert click coordinates to (-0.5, 0.5) range
        norm_x, norm_y = normalize_click(x - start_x, y - start_y, img_w, img_h)
        set_annot(img_path, a_name, [round(norm_x, 4), round(norm_y, 4)])
        dialog['saved'] = True

def process_images(image_dir):
    files  = sorted(os.listdir(image_dir))
    index  = 0
    images = []
    #random.shuffle(files)
    for filename in files:
        if filename.lower().endswith((".png")):
            img_path  = os.path.join(image_dir, filename)
            json_path = os.path.splitext(img_path)[0] + ".json"
            if not review and os.path.exists(json_path):
                continue
            images.append(img_path)
    
    while index < len(images):
        img_path  = images[index]
        img       = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img       = np.expand_dims(img, axis=-1)
        img_h, img_w = img.shape[:2]
        global dialog, start_x, start_y, canvas_w, canvas_h
        dialog    = {}
        canvas_w  = img_w * 2
        canvas_h  = img_h * 2
        canvas    = np.zeros((canvas_h, canvas_w, 1), dtype=np.uint8)
        start_x   = (canvas_w - img_w) // 2
        start_y   = (canvas_h - img_h) // 2
        canvas[start_y:start_y + img_h, start_x:start_x + img_w] = img
        print(f"showing: {img_path} - annotating {a_name}")
        title     = f'hyperspace:annotate - {a_name}'
        cv2.imshow(title, canvas)
        cv2.setMouseCallback(title, on_click, (img_path, img_w, img_h))

        while True:
            key = cv2.waitKey(20) & 0xFF
            if "saved" in dialog:
                cv2.setMouseCallback(title, lambda *args: None)
                break
            else:
                if key == 27:
                    return
                elif key == ord('d') or key == 83:  # Next
                    cv2.setMouseCallback(title, lambda *args: None)
                    break
                elif key == ord('a') or key == 81:  # Prev
                    cv2.setMouseCallback(title, lambda *args: None)
                    index -= 2
                    break
        
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        index = max(index + 1, 0)

process_images(IMAGE_DIR)
cv2.destroyAllWindows()
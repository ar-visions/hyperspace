#!/bin/env python3
import os
import json
import cv2

# Directory containing images
IMAGE_DIR    = "sessions/wkjykq"  # <-- Change this to your folder
LABELS_DIR   = IMAGE_DIR  # JSON files will be saved in the same directory
dialog = {}

def normalize_click(x, y, img_w, img_h):
    """Converts absolute click (x, y) to a normalized (-0.5 to 0.5) range."""
    return [(x - img_w / 2) / img_w, (y - img_h / 2) / img_h]

def on_click(event, x, y, flags, param):
    """Handles mouse click event, normalizes coordinates, and saves to JSON."""
    if event == cv2.EVENT_LBUTTONDOWN:
        img_path, img_w, img_h = param  # Unpack params

        # Convert click coordinates to (-0.5, 0.5) range
        norm_x, norm_y = normalize_click(x, y, img_w, img_h)

        # JSON filename
        json_path = os.path.splitext(img_path)[0] + ".json"

        # Create JSON structure
        json_data = {"labels": [{"eye-center": [round(norm_x, 4), round(norm_y, 4)]}]}

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        #print(f"Saved: {json_path} -> {json_data}")
        dialog['saved'] = True
        # Close image window after clicking
        #cv2.destroyAllWindows()

def process_images(image_dir):
    """Iterates over images and captures user clicks using OpenCV."""
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)

            # Load image
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]

            # Display instructions
            print(f"Showing: {filename} - Click on the eye center!")
            global dialog
            dialog = {}
            # Show image and capture click
            cv2.imshow("Select Eye Center (Click to Save)", img)
            cv2.setMouseCallback("Select Eye Center (Click to Save)", on_click, (img_path, img_w, img_h))
            # Wait until the user clicks (polling every 20ms)
            while True:
                key = cv2.waitKey(20) & 0xFF
                if "saved" in dialog:
                    break
            
            # Wait until a key is pressed
            cv2.destroyAllWindows()
            cv2.waitKey(10)


# Run the script
process_images(IMAGE_DIR)

cv2.destroyAllWindows()

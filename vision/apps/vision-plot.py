#!/usr/bin/env python3
# pip install torch torchvision opencv-python numpy requests pillow
# ----------------------------------------------------------
import  cv2
import  os
import  numpy as np
import  requests
import  random
import  string
import  time
from    io  import BytesIO
from    PIL import Image
from    tkinter import Tk
from    playsound import playsound

def get_screen_resolution():
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

screen_width, screen_height = get_screen_resolution()
print(f'screen = {screen_width}, {screen_height}')

def random_bg_color():
    g = random.randint(0, 64)
    return np.full((screen_height, screen_width), g, dtype=np.uint8)

def generate_hash_id():
    """Generate a random 6-letter hash ID"""
    return ''.join(random.choices(string.ascii_lowercase, k=6))

# logitech brio is a good device, /dev/video6
camera_width, camera_height = 340, 340
title        = "hyperspace:vision trainer"
frame_count  = 0
pip_scale    = 0.5  # Scale the PiP window to 10% of the background
pip_pad      = 0.1
pip_width, pip_height = int(camera_width * pip_scale), int(camera_height * pip_scale)
pip_x, pip_y = screen_width - pip_width - int(camera_width * pip_pad), screen_height - pip_height - int(camera_width * pip_pad)  # Bottom-right with padding
session      = generate_hash_id()
images2      = []
bright2      = []

os.makedirs(f'sessions/{session}', exist_ok=True)

cap2 = cv2.VideoCapture(6)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
if not cap2.isOpened():
    print("error: could not open camera 2")
    exit(2)

cv2.namedWindow      (title, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
background = random_bg_color()
cv2.imshow(title, background)

# rectangle: head here (center of rotation = your eye median); not the lower middle of your brain (who even knows where this is)
# circle: look at this

zone      = 3
zone_cm   = [ 10, 30, 50 ]
z_max     = 50
head_x    = -1
head_y    = -1
head_z    = 0
eye_x     = -1
eye_y     = -1
pcount    = 4
pindex    = 0
key       = 0
eyes_only = True

while True:
    # refresh background image every 3 seconds
    if head_x == -1 or (key == 32): # we should also support audio-based capture
        playsound('beep.wav')
        if head_x != -1:
            filename2 = f"sessions/{session}/f2_{frame_count:04d}_{eye_x/screen_width:.4f}_{eye_y/screen_height:.4f}_{head_x/screen_width:.4f}_{head_y/screen_height:.4f}_{head_z/z_max:.4f}.png"
            cv2.imwrite(filename2, selected, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"saved: {filename2}")

        if eyes_only:
            head_x = screen_width // 2
            head_y = screen_height // 2
            eye_x  = random.randint(0,   screen_width)
            eye_y  = random.randint(0,   screen_height)
            zone   = 1
            head_z = zone_cm[zone]
        else:
            pindex += 1
            if pindex >= pcount:
                pindex = 0
            if head_x == -1 or pindex == 0:
                zone -= 1
                if zone < 0: zone = 2
                if head_x == -1 or zone == 2:
                    pad    = screen_width / 16
                    head_x = random.randint(pad, screen_width - pad * 2)
                    head_y = random.randint(pad, screen_height - pad * 2)
                    eye_x  = random.randint(0,   screen_width)
                    eye_y  = random.randint(0,   screen_height)
                head_z = zone_cm[zone]  # Radius in pixels

    # capture frame from webcam, and select the best and brightest
    ret2, frame2 = cap2.read()
    if not ret2:
        print("error: could not read frame (camera 2)")
        exit(2)
    gray_image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray_image2)
    images2.insert(0, gray_image2)
    bright2.insert(0, current_brightness)
    if len(images2) > 2:
        images2.pop()
        bright2.pop()
    best_bright = current_brightness
    selected = gray_image2
    index = 0
    for b_value in bright2:
        if b_value > best_bright:
            selected    = images2[index]
            best_bright = b_value
        index += 1

    # display reduced frame; not large enough to cause too much noise
    pip_frame2    = cv2.resize(selected, (pip_width, pip_height))
    blended_frame = background.copy()  # Copy background to avoid modifying original
    blended_frame[pip_y:pip_y+pip_height, pip_x:pip_x+pip_width]   = pip_frame2

    half = head_z // 2 * 2
    cv2.rectangle(blended_frame, (head_x - half, head_y - half * 2), (head_x + half, head_y + half * 2), 255, 3)  # 255=white, 1=thickness
    cv2.circle(blended_frame, (eye_x, eye_y), 4, 255, 2)  # -1 thickness fills the circle

    # we want to draw it here ... rx, ry, rr will be set (radius, and x y) in screen coords
    cv2.imshow(title, blended_frame)

    frame_count += 1
    key = cv2.waitKey(10) & 0xFF

    if key == 27: break # esc

cv2.destroyAllWindows()
cap2.release()
#cap4.release()
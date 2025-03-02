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
from    io          import BytesIO
from    PIL         import Image
from    tkinter     import Tk
from    playsound   import playsound
import  argparse
import  sounddevice as sd

mic_level = 0

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    rms = np.sqrt(np.mean(np.square(indata)))
    global mic_level
    mic_level = min(1.0, rms * 10)  # Scale it to 0 - 1

def get_screen_resolution():
    root          = Tk()
    screen_width  = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

parser = argparse.ArgumentParser(description="Annotate images with click-based labels.")
parser.add_argument("-s", "--session",     required=True, help="session data (name of session; as named in cwd/sessions dir)")
parser.add_argument("-f", "--debug",       action="store_true", help="debug mode (cannot record)")

args   = parser.parse_args()
debug        = args.debug
screen_width, screen_height = get_screen_resolution()
# logitech brio is a good device, /dev/video6
camera_width, camera_height = 340, 340
title        = "hyperspace:record"
frame_count  = 0
pip_scale    = 0.5  # Scale the PiP window to 10% of the background
pip_pad      = 0.5
pip_width, pip_height = int(camera_width * pip_scale), int(camera_height * pip_scale)
pip_x, pip_y = screen_width - pip_width - int(camera_width * pip_pad), screen_height - pip_height - int(camera_width * pip_pad)  # Bottom-right with padding
session      = args.session

def random_bg_color():
    g = random.randint(0, 64)
    return np.full((screen_height, screen_width), g, dtype=np.uint8)

def generate_hash_id():
    return ''.join(random.choices(string.ascii_lowercase, k=6))

os.makedirs(f'sessions/{session}', exist_ok=True)

# monitor microphone levels
sample_rate = 44100  # standard audio sample rate (Hz)
block_size  = 2048   # number of frames per block
stream      = sd.InputStream(callback=audio_callback, 
                        channels=1,
                        samplerate=sample_rate,
                        blocksize=block_size)
stream.start()  # Start listening to the microphone

if not debug:
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

class cam:
    def __init__(self, device_id):
        self.device_id = device_id
        self.cap = None
        self.images = []
        self.bright = []
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        # Disable all automatic adjustments
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)  # 0.25 is manual mode (OFF)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)        # Disable autofocus
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # Disable auto white balance
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)       # Set fixed brightness
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0)        # Set fixed contrast
        self.cap.set(cv2.CAP_PROP_GAIN, 0)             # Set manual gain
        
        # Set exposure if provided
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
        
        if not self.cap.isOpened():
            print(f"error: could not open (camera {self.device_id})")
            exit(2)

    def read(self):
        # capture frame from webcam, and select the best and brightest
        ret, frame = self.cap.read()
        if not ret:
            print(f"error: could not read frame (camera {self.device_id})")
            exit(2)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray_image)
        self.images.insert(0, gray_image)
        self.bright.insert(0, current_brightness)
        if len(self.images) > 2:
            self.images.pop()
            self.bright.pop()
        best_bright = current_brightness
        selected = gray_image
        index = 0
        for b_value in self.bright:
            if b_value > best_bright:
                selected = self.images[index]
                best_bright = b_value
            index += 1

        # display reduced frame; not large enough to cause too much noise
        pip_frame = cv2.resize(selected, (pip_width, pip_height))
        return selected, pip_frame

cam2      = cam(2) # top
cam6      = cam(6) # bottom (stOP!.... bottom one!)
                   # i need glue for this bottom one; or call-upon someone to help me mount it properly.
selected  = False
selected2 = None
selected6 = None

##
# we want all of this in C with a variable set of cameras that the user must bind;
# not so hard to have a thumbnail and a option for each
#
# any overwriting camera would null any users duplicating it; thats better than toggling other off to make the other visible
# no confirmation dialogs or anything silly. trinity needs in it.  theres no way im making that separate from trinity
# trinity = UX (father, quantum membrane descriptor language to outter event horizon), 
#           3D (son,    the physical realm, continuously encoding new entanglements in timescapes),
#           AR (spirit; voice, and body -- something to visualize more dimensions of data and move with us on 
#                       corrective-counter-planar projection)
# 
##

def write(id, image):
    filename = f"sessions/{session}/f{id}_{session}-{frame_count:04d}_{eye_x/screen_width:.4f}_{eye_y/screen_height:.4f}_{head_x/screen_width:.4f}_{head_y/screen_height:.4f}_{head_z/z_max:.4f}.png"
    cv2.imwrite(filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"saved: {filename}")

while True:
    # refresh background image every 3 seconds
    if not debug and selected and (head_x == -1 or (key == 32)): # we should also support audio-based capture
        if head_x != -1:
            playsound('beep.wav')
            write(2, selected2)
            write(6, selected6)

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

    
    blended_frame = background.copy()

    # indicate which ones are selected, through here if we need
    selected = True
    # camera 2 (top)
    selected2, pip_frame2 = cam2.read()
    blended_frame[pip_y-pip_height:pip_y, pip_x:pip_x+pip_width] = pip_frame2
    # camera 6 (bottom; ... brick, not hit back)
    selected6, pip_frame6 = cam6.read()
    blended_frame[pip_y:pip_y+pip_height, pip_x:pip_x+pip_width] = pip_frame6

    half = head_z // 2 * 2
    cv2.rectangle(blended_frame, (head_x - half, head_y - half * 2), (head_x + half, head_y + half * 2), 255, 3)  # 255=white, 1=thickness
    cv2.circle(blended_frame, (eye_x, eye_y), 4, 255, 2)  # -1 thickness fills the circle
    cv2.imshow(title, blended_frame)
    frame_count += 1
    
    print(f'mic: {mic_level}')
    key          = cv2.waitKey(10) & 0xFF
    if key == 27: break # esc

cv2.destroyAllWindows()

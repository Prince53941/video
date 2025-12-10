import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from math import hypot

# Try importing PyCaw for Volume Control (Windows primarily)
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    pycaw_available = True
except ImportError:
    pycaw_available = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gesture Control AI", page_icon="🖐️", layout="centered")

st.title("🖐️ AI Hand Gesture Controller")
st.markdown("""
Control your PC using hand gestures! 
1. **Show your hand** to the camera.
2. **Pinch** your Thumb and Index finger to adjust levels.
3. Select **Volume** or **Brightness** mode below.
""")

# --- 1. SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- 2. SETUP VOLUME CONTROL (PyCaw) ---
volume = None
volRange = [0, 0]
if pycaw_available:
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        volRange = volume.GetVolumeRange() # usually (-65.25, 0.0)
    except:
        pycaw_available = False

# --- 3. HELPER FUNCTIONS ---
def get_distance(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])

# --- 4. STREAMLIT UI LAYOUT ---
mode = st.radio("Select Control Mode:", ("System Volume", "Screen Brightness"), horizontal=True)
run = st.checkbox("Start Camera", value=False)
frame_placeholder = st.empty()

# --- 5. MAIN LOOP ---
if run:
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened
    if not cap.isOpened():
        st.error("Could not access the camera.")
    
    while run:
        success, img = cap.read()
        if not success:
            st.warning("Camera disconnected.")
            break

        # Convert image for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get ID and coordinates for landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

        if len(lmList) != 0:
            # Tip of Thumb (ID 4) and Index Finger (ID 8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            
            # Midpoint (for visuals)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw points and line
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Calculate Distance
            length = get_distance((x1, y1), (x2, y2))
            
            # Hand range: approx 30 (closed) to 250 (open)
            # Map this range to 0-100 or Volume Range
            
            # --- MODE: VOLUME ---
            if mode == "System Volume":
                if pycaw_available:
                    # Map hand range (30-200) to Volume Range (min-max)
                    vol = np.interp(length, [30, 200], [volRange[0], volRange[1]])
                    volume.SetMasterVolumeLevel(vol, None)
                    
                    # Visual Bar
                    volBar = np.interp(length, [30, 200], [400, 150])
                    volPer = np.interp(length, [30, 200], [0, 100])
                    
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                else:
                    cv2.putText(img, "PyCaw not installed/supported", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- MODE: BRIGHTNESS ---
            elif mode == "Screen Brightness":
                # Map hand range (30-200) to Brightness (0-100)
                bright = np.interp(length, [30, 200], [0, 100])
                try:
                    sbc.set_brightness(int(bright))
                except:
                    pass # Ignore errors on unsupported displays
                
                cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
                cv2.rectangle(img, (50, int(np.interp(bright, [0, 100], [400, 150]))), (85, 400), (255, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(bright)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

        # Update Streamlit Frame
        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()
else:
    st.info("Check 'Start Camera' to begin.")

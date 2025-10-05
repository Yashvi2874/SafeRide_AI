import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import os

# Initialize YOLO model for object detection
# Check for custom trained model first, fallback to default
model_path = "yolov8n_custom.pt" if os.path.exists("yolov8n_custom.pt") else "yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"
model = YOLO(model_path)
model.overrides['verbose'] = False
model.overrides['imgsz'] = 320

# Define object classes for detection
object_classes = ["cell phone", "book", "laptop"]
fidget_objects = ["cell phone", "book"]
phone_aliases = ["cell phone", "phone", "mobile phone"]

# Initialize MediaPipe components
try:
    mp_face_mesh = getattr(mp.solutions, 'face_mesh', None)
    mp_hands = getattr(mp.solutions, 'hands', None)
    mp_drawing = getattr(mp.solutions, 'drawing_utils', None)
    MEDIAPIPE_AVAILABLE = mp_face_mesh is not None and mp_hands is not None and mp_drawing is not None
except Exception:
    mp_face_mesh = None
    mp_hands = None
    mp_drawing = None
    MEDIAPIPE_AVAILABLE = False

if MEDIAPIPE_AVAILABLE and mp_face_mesh and mp_hands:
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0
    )
else:
    face_mesh = None
    hands = None

# Calculate eye aspect ratio for blink detection
def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [np.array(landmarks[i]) for i in eye_indices]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    EAR = (A + B) / (2.0 * C)
    return EAR

# Calculate mouth aspect ratio for yawn detection
def mouth_aspect_ratio(landmarks, mouth_indices):
    try:
        mouth_points = [np.array(landmarks[i]) for i in mouth_indices]
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])
        B = np.linalg.norm(mouth_points[3] - mouth_points[9])
        C = np.linalg.norm(mouth_points[4] - mouth_points[8])
        D = np.linalg.norm(mouth_points[0] - mouth_points[6])
        if D == 0:
            return 0.0
        MAR = (A + B + C) / (3.0 * D)
        return MAR
    except Exception:
        return 0.0

# Detect if person is speaking or eating to avoid false yawning positives
def is_speaking_or_eating(landmarks):
    try:
        mouth_points = [np.array(landmarks[i]) for i in MOUTH]
        mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
        mouth_height = np.linalg.norm(mouth_points[2] - mouth_points[10])
        if mouth_width > 0 and mouth_height > 0:
            width_height_ratio = mouth_width / mouth_height
            return width_height_ratio < 3.0
        return False
    except Exception:
        return False

# Detect gaze direction
def get_gaze_direction(landmarks, w, h):
    try:
        left_eye = np.array(landmarks[133])
        right_eye = np.array(landmarks[362])
        nose_tip = np.array(landmarks[1])
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        face_center_x = w / 2
        face_center_y = h / 2
        x_diff = abs(eye_center_x - face_center_x)
        y_diff = abs(eye_center_y - face_center_y)
        x_threshold = w * 0.08
        y_threshold = h * 0.08
        looking_away = (x_diff > x_threshold or y_diff > y_threshold)
        return not looking_away
    except Exception:
        return True

# Facial landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Hand near face detection
def is_hand_near_face(hand_landmarks, face_landmarks, threshold=0.15):
    try:
        face_x = [lm[0] for lm in face_landmarks]
        face_y = [lm[1] for lm in face_landmarks]
        face_left, face_right = min(face_x), max(face_x)
        face_top, face_bottom = min(face_y), max(face_y)
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        face_left -= face_width * threshold
        face_right += face_width * threshold
        face_top -= face_height * threshold
        face_bottom += face_height * threshold
        for lm in hand_landmarks:
            if (face_left <= lm[0] <= face_right and 
                face_top <= lm[1] <= face_bottom):
                return True
        return False
    except Exception:
        return False

# Check if hand is holding an object
def is_object_in_hand(hand_landmarks, object_bbox, threshold=100):
    try:
        x1, y1, x2, y2 = object_bbox
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Check multiple points on the hand (fingertips and palm center)
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky fingertips
        palm_center = 0  # Palm center landmark
        
        # Check if any fingertip or palm center is close to object
        for idx in fingertips + [palm_center]:
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                distance = np.sqrt((lm[0] - obj_center_x)**2 + (lm[1] - obj_center_y)**2)
                if distance < threshold:
                    return True
        return False
    except Exception:
        return False

# Enhanced hand-phone interaction detection
def is_phone_in_hand(hand_landmarks, phone_bbox, frame_width, frame_height):
    """
    Enhanced detection of phone in hand with multiple criteria
    """
    try:
        x1, y1, x2, y2 = phone_bbox
        phone_center_x = (x1 + x2) / 2
        phone_center_y = (y1 + y2) / 2
        phone_width = x2 - x1
        phone_height = y2 - y1
        
        # Check if phone is in a reasonable position (not too high or too low)
        if phone_center_y < frame_height * 0.2 or phone_center_y > frame_height * 0.8:
            return False
            
        # Check multiple hand points
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky fingertips
        palm_center = 0  # Palm center landmark
        
        # Adaptive threshold based on phone size
        adaptive_threshold = max(phone_width, phone_height) * 0.8
        
        # Check if any hand point is close to phone
        for idx in fingertips + [palm_center]:
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                distance = np.sqrt((lm[0] - phone_center_x)**2 + (lm[1] - phone_center_y)**2)
                if distance < adaptive_threshold:
                    return True
        return False
    except Exception:
        return False

# Calculate hand movement
def calculate_hand_movement(current_hands, previous_hands, threshold=0.02):
    if previous_hands is None or current_hands is None:
        return False
    if len(current_hands) != len(previous_hands):
        return False
    try:
        total_movement = 0
        num_landmarks = 0
        for curr_hand, prev_hand in zip(current_hands, previous_hands):
            if len(curr_hand) != len(prev_hand):
                continue
            for i in range(0, len(curr_hand), 3):
                curr_lm = curr_hand[i]
                prev_lm = prev_hand[i]
                movement = np.linalg.norm(np.array(curr_lm) - np.array(prev_lm))
                total_movement += movement
                num_landmarks += 1
        if num_landmarks == 0:
            return False
        avg_movement = total_movement / num_landmarks
        return avg_movement > threshold
    except Exception:
        return False

# Tracking variables
attention_scores = deque(maxlen=10)
previous_hand_positions = None
fidget_start_time = None
fidget_active = False
fidget_penalty_active = False
playing_with_object = False

# Drowsiness detection parameters
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 30
EAR_WARNING_FRAMES = 8
ear_counter = 0
drowsy = False
drowsy_warning = False

# Yawning detection parameters
YAWN_THRESHOLD = 0.7
YAWN_CONSEC_FRAMES = 25
YAWN_RECOVERY_RATE = 3
mar_counter = 0
yawning = False

# Phone detection state
phone_detected = False
hand_on_phone = False
phone_usage_counter = 0  # Counter for persistent phone usage detection
PHONE_USAGE_THRESHOLD = 2  # Further reduced frames needed to confirm phone usage

# Additional phone detection variables
phone_confidence_history = deque(maxlen=5)  # Track recent phone confidences
last_phone_detection_time = 0
PHONE_DETECTION_TIMEOUT = 2.0  # seconds

# Enhanced phone detection parameters
PHONE_CONFIDENCE_THRESHOLD = 0.15  # Lower threshold for better detection
MIN_PHONE_SIZE = 50  # Minimum phone size in pixels
MAX_PHONE_SIZE = 300  # Maximum phone size in pixels

# Attention recovery system
RECOVERY_RATE = 0.8
recovery_mode = False
recovery_score = 0

# Display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Main detection function
def detect_attention_and_drowsiness(frame, speech_recognizer=None):
    global attention_scores, previous_hand_positions, fidget_start_time
    global fidget_active, fidget_penalty_active, playing_with_object
    global ear_counter, mar_counter, drowsy, drowsy_warning, recovery_mode, recovery_score
    global phone_detected, hand_on_phone, phone_usage_counter, yawning, last_phone_detection_time
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    h, w, _ = frame.shape

    # Object detection with YOLO
    yolo_results = model(frame, stream=False, verbose=False)
    distractions = []
    object_bboxes = {}
    cell_phone_detected = False
    phone_confidences = []  # Track phone detection confidences
    current_time = time.time()

    for r in yolo_results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            confidence = float(box.conf[0])
            
            # Enhanced phone detection
            if cls == "cell phone":
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                # Check if phone size is reasonable
                if MIN_PHONE_SIZE <= width <= MAX_PHONE_SIZE and MIN_PHONE_SIZE <= height <= MAX_PHONE_SIZE:
                    phone_confidences.append(confidence)
                    # Use adaptive threshold based on recent detections
                    threshold = PHONE_CONFIDENCE_THRESHOLD
                    if len(phone_confidence_history) > 0:
                        avg_confidence = sum(phone_confidence_history) / len(phone_confidence_history)
                        threshold = max(PHONE_CONFIDENCE_THRESHOLD, avg_confidence * 0.7)
                    
                    if confidence > threshold:
                        cell_phone_detected = True
                        phone_detected = True
                        distractions.append(cls)
                        object_bboxes[cls] = (x1, y1, x2, y2)
                        
                        # Update confidence history
                        phone_confidence_history.append(confidence)
                        last_phone_detection_time = current_time
            elif cls in object_classes and confidence > 0.3:
                distractions.append(cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_bboxes[cls] = (x1, y1, x2, y2)
    
    # More responsive phone detection reset with timeout
    if not cell_phone_detected:
        # Only reset if no phone detected for a while or confidence is very low
        time_since_last_detection = current_time - last_phone_detection_time
        if time_since_last_detection > PHONE_DETECTION_TIMEOUT or len(phone_confidences) == 0:
            phone_detected = False
            phone_usage_counter = max(0, phone_usage_counter - 2)  # Faster decay
        hand_on_phone = (phone_usage_counter >= PHONE_USAGE_THRESHOLD)
    else:
        # Update last detection time
        last_phone_detection_time = current_time

    # Face mesh detection
    face_detected = False
    eyes_open = False
    looking_at_screen = False
    EAR_avg = 0
    MAR_avg = 0
    face_landmarks_list = None

    if face_mesh and MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            face_detected = True
            face_landmarks = face_results.multi_face_landmarks[0]
            face_landmarks_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
            leftEAR = eye_aspect_ratio(face_landmarks_list, LEFT_EYE)
            rightEAR = eye_aspect_ratio(face_landmarks_list, RIGHT_EYE)
            EAR_avg = (leftEAR + rightEAR) / 2.0
            eyes_open = EAR_avg > EAR_THRESHOLD
            MAR_avg = mouth_aspect_ratio(face_landmarks_list, MOUTH)
            speaking_or_eating = is_speaking_or_eating(face_landmarks_list)
            
            if face_detected and eyes_open and MAR_avg > YAWN_THRESHOLD and not speaking_or_eating:
                mar_counter += 1
            else:
                mar_counter = max(0, mar_counter - YAWN_RECOVERY_RATE)
            
            yawning = mar_counter >= YAWN_CONSEC_FRAMES
            looking_at_screen = get_gaze_direction(face_landmarks_list, w, h)
            
            # Draw face mesh landmarks
            if mp_drawing and hasattr(mp_drawing, 'draw_landmarks'):
                try:
                    mp_drawing.draw_landmarks(
                        frame, 
                        face_landmarks, 
                        getattr(mp_face_mesh, 'FACEMESH_CONTOURS', None),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                    )
                except Exception:
                    pass
    
    # Drowsiness detection
    if face_detected:
        if not eyes_open:
            ear_counter += 1
        else:
            ear_counter = max(0, ear_counter - 2)

        drowsy_warning = (ear_counter >= EAR_WARNING_FRAMES and ear_counter < EAR_CONSEC_FRAMES)
        drowsy = (ear_counter >= EAR_CONSEC_FRAMES)
    else:
        ear_counter = 0
        drowsy_warning = False
        drowsy = False
        mar_counter = 0
        yawning = False

    # Hand detection
    hands_detected = False
    hand_near_face = False
    current_hand_positions = []

    if hands and MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        
        if hand_results.multi_hand_landmarks:
            hands_detected = True
            hand_on_phone_current = False
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_lm_list = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                current_hand_positions.append(hand_lm_list)
                
                if face_landmarks_list:
                    hand_near_face = is_hand_near_face(hand_lm_list, face_landmarks_list)
                    if hand_near_face:
                        break
                
                # Check if hand is on phone (only if phone is detected)
                if phone_detected and object_bboxes.get("cell phone"):
                    # Try both methods for better detection
                    method1 = is_object_in_hand(hand_lm_list, object_bboxes["cell phone"])
                    method2 = is_phone_in_hand(hand_lm_list, object_bboxes["cell phone"], w, h)
                    if method1 or method2:
                        hand_on_phone_current = True
                
                # Draw hand landmarks
                if mp_drawing and hasattr(mp_drawing, 'draw_landmarks'):
                    try:
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            getattr(mp_hands, 'HAND_CONNECTIONS', None),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                        )
                    except Exception:
                        pass
            
            # More responsive hand-on-phone state update
            if hand_on_phone_current:
                phone_usage_counter = min(phone_usage_counter + 1, PHONE_USAGE_THRESHOLD + 2)  # Allow slightly higher counter
            else:
                phone_usage_counter = max(0, phone_usage_counter - 1)
            
            # More responsive hand_on_phone detection
            hand_on_phone = (phone_usage_counter >= PHONE_USAGE_THRESHOLD) or (hand_on_phone and phone_usage_counter >= 1)
        else:
            # Decay faster when no hands detected
            phone_usage_counter = max(0, phone_usage_counter - 2)
            # Keep hand_on_phone state for a short time to avoid flickering
            if phone_usage_counter < PHONE_USAGE_THRESHOLD - 1:
                hand_on_phone = False
    else:
        # Decay even faster when no hands detected
        phone_usage_counter = max(0, phone_usage_counter - 3)
        hand_on_phone = False

    # Attention scoring
    score = 50

    if face_detected:
        score += 25
    if eyes_open:
        score += 25
    if looking_at_screen:
        score += 20

    if drowsy:
        score -= 40
    elif drowsy_warning:
        score -= 20

    if face_detected and not looking_at_screen:
        score -= 25

    for obj in distractions:
        if obj == "cell phone":
            score -= 35
        elif obj == "laptop":
            score -= 25
        elif obj == "book":
            score -= 20

    if hand_near_face:
        score -= 15
        
    # Apply phone usage penalty only when hand is confirmed on phone
    if hand_on_phone:
        score -= 30
        
    if yawning:
        score -= 25

    attention = max(0, min(100, score))

    # Temporal smoothing
    attention_scores.append(attention)
    smooth_attention = sum(attention_scores) / len(attention_scores)

    # Display information
    color = (0, 255, 0) if smooth_attention > 70 else (0, 255, 255) if smooth_attention > 40 else (0, 0, 255)
    cv2.putText(frame, f"Attention: {smooth_attention:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    y_pos = 60
    if drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
    elif drowsy_warning:
        cv2.putText(frame, "EYES CLOSING...", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y_pos += 30
    
    if yawning:
        cv2.putText(frame, "SLEEPINESS ALERT (YAWNING)", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_pos += 30
    
    # Show phone usage alert with better visibility
    if phone_detected and hand_on_phone:
        cv2.putText(frame, "LOW ATTENTION (ON CALL)", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        y_pos += 30
    elif phone_detected:
        cv2.putText(frame, "PHONE DETECTED", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # Show confidence levels for debugging
        if phone_confidences:
            conf_text = f"Conf: {', '.join([f'{c:.2f}' for c in phone_confidences[:3]])}"
            cv2.putText(frame, conf_text, (10, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Show phone size for debugging
            if "cell phone" in object_bboxes:
                x1, y1, x2, y2 = object_bboxes["cell phone"]
                size_text = f"Size: {x2-x1}x{y2-y1}"
                cv2.putText(frame, size_text, (10, y_pos + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw object detection boxes
    for obj_name, bbox in object_bboxes.items():
        x1, y1, x2, y2 = bbox
        if obj_name == "cell phone":
            color = (0, 0, 255) if hand_on_phone else (255, 255, 0)  # Red if in hand, yellow if detected
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Add confidence text
            if phone_confidences:
                conf_text = f"Phone: {max(phone_confidences):.2f}"
                cv2.putText(frame, conf_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)

    # Determine alert type for GUI
    if drowsy:
        drowsiness_alert = "DROWSY"
    elif drowsy_warning:
        drowsiness_alert = "WARNING"
    elif yawning:
        drowsiness_alert = "SLEEPY"
    elif phone_detected:
        drowsiness_alert = "LOW_ATTENTION"
    else:
        drowsiness_alert = "NONE"
    
    return frame, smooth_attention, drowsiness_alert, yawning
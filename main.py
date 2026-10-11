import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize mediapipe hands and drawing ultilities
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)

# Create the detector instance
detector = vision.HandLandmarker.create_from_options(options)

# open default webcam
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

print("Press q in the video window to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab webcam")
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR frame to RGB for Mediapipe
    mp_image = mp.Image(
      image_format=mp.ImageFormat.SRGB,
      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )

    # Increment timestamp
    frame_timestamp_ms += int(1000/30) # 30 fps
    
    # Run landmark detection
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Process and draw results if hands are detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # Highlight index finger tip with a green circle
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

    # Display the live window
    cv2.imshow("Gesture Control - MediaPipe Tasks", frame)

    # Exit loop when q is pressed
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()






#THIS IS TO FIND PEOPLE IN A STATIC IMAGE AKA NOT LIVE

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Replace 'IMAGE_ADDRESS' with the address of the image you want to use
IMAGE_ADDRESS = r"Put here what you wanna find"

# Load the image
image = cv2.imread(IMAGE_ADDRESS)

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
        print("No faces found.")
    else:
        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)

            # Add your name underneath the detection
            cv2.putText(annotated_image, "This a Person", (int(detection.location_data.relative_bounding_box.xmin * image.shape[1]),
                                                 int(detection.location_data.relative_bounding_box.ymin * image.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Face Detection', annotated_image)
        cv2.waitKey(0)

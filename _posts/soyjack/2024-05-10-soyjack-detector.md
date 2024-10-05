---
title: "Soyjack Detector"
date: 2024-05-10 08:00:00 +00:00
tags: [coding, ai]
toc: false
---

<figure style="text-align: center;">
  <img src="/assets/img/soyjack/soy_detected.png" alt="soy">
</figure>

# Why ? 

Because <a href="https://x.com/yacineMTB/status/1841954975745757687">it's funny</a>

# How ?

The whole thing is actually quite simple with <a href="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker">Google Face Landmarker</a>, which is basically 3 models packaged together: 
- Face detection model: detects the presence of faces with a few key facial landmarks.
- Face mesh model: adds a complete mapping of the face. The model outputs an estimate of 478 3-dimensional face landmarks.
- Blendshape prediction model: receives output from the face mesh model predicts 52 blendshape scores, which are coefficients representing facial different expressions.

We just run this on a video with a secret soyjack algorithm and voilà, we can detect soyjacks in real time (from your webcam !!!)

# Code

Need to install `mediapipe` and download the packaged models:

```bash
pip install -q mediapipe
```
```bash
!wget -O face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Then we can define some helper function to nicely draw the detected landmarks and face mesh:

```python

import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
    
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
    
    # Draw the face landmarks and mesh
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

    return annotated_image
```

The actual inference model can be simply created from the downloaded model like so : 

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
```
The main loop is quite simple, we ingest frames from the webcam feed and run the model on each frame:

```python

import cv2
import mediapipe as mp

def setup_webcam(camera_id=0, width=800, height=800):
    """
    Initialize the webcam with specified parameters
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap


def add_text_to_frame(frame, text, position=None):
    height, width = frame.shape[:2]
    if position is None:
        position = (width - 500, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255,192,203)
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, 
                 (position[0] - 5, position[1] - text_height - 5),
                 (position[0] + text_width + 5, position[1] + 5),
                 (0, 0, 0),
                 -1)
    cv2.putText(frame,
                text,
                position,
                font,
                font_scale,
                font_color,
                thickness)
    
    return frame


def main(privacy=False):
    cap = setup_webcam()
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert the webcam frame into correct format and run inference
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            
            # If turned on, only display the detected face landmarks
            if privacy:
                frame = np.zeros(frame.shape)
            
            # Draw detected landmarks on the webcam frame
            annotated_image = draw_landmarks_on_image(frame, detection_result)
            
            # Top secret algorithm to detect a soyface based on the model outputs
            text = "NO SOYJACK DETECTED !!!"
            jaw_open = 0
            eyes_wide = 0
            brows_up = 0
            smile_wide = 0
            try:
                for d in detection_result.face_blendshapes[0]:
                    if d.category_name == 'jawOpen':
                        jaw_open = d.score
                    elif d.category_name in ['eyeWideLeft', 'eyeWideRight']:
                        eyes_wide = max(eyes_wide, d.score)
                    elif d.category_name in ['browOuterUpLeft', 'browOuterUpRight']:
                        brows_up = max(brows_up, d.score)
                    elif d.category_name in ['mouthSmileLeft', 'mouthSmileRight']:
                        smile_wide = max(smile_wide, d.score)
                # Top secret
                soyjack_score = (eyes_wide * 0.3 + brows_up * 0.4 + smile_wide * 0.3)
                if jaw_open >= 0.5 or soyjack_score >= 0.5:
                    soyjack_score = max(jaw_open, soyjack_score)
                    text = f"SOYJACK DETECTED !!! (soyprob : {soyjack_score:.2%})"
                
                # Display the soyscore if detected
                annotated_image = add_text_to_frame(annotated_image, text)
                annotated_image = add_text_to_frame(annotated_image, f"PRIVACY MODE : {privacy}", position=(annotated_image.shape[1] - 250,
                                                                                                            annotated_image.shape[0] - 50))
                cv2.imshow('Webcam Feed', annotated_image)
            
            # Might fail in case of extreme occlusion etc, just ignore the frame
            except:
                print("NO LANDMARKS DETECTED")
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break          
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # If set to True, only the detected landmarks will be displayed instead of the entire webcam feed 
    USE_PRIVACY_MODE = False
    main(privacy=USE_PRIVACY_MODE)
```

That's pretty much it. If I'm not lazy I'll package it into an app and add a simple frontend and maybe more options. Happy soyjacking !

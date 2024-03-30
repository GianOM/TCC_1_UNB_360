import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the face landmarker model bundle.
      num_faces: Max number of faces that can be detected by the landmarker.
      min_face_detection_confidence: The minimum confidence score for face
        detection to be considered successful.
      min_face_presence_confidence: The minimum confidence score of face
        presence score in the face landmark detection.
      min_tracking_confidence: The minimum confidence score for the face
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Label box parameters
    label_background_color = (255, 255, 255)  # White
    label_padding_width = 1500  # pixels

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the face landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=vision.RunningMode.LIVE_STREAM,
                                            num_faces=num_faces,
                                            min_face_detection_confidence=min_face_detection_confidence,
                                            min_face_presence_confidence=min_face_presence_confidence,
                                            min_tracking_confidence=min_tracking_confidence,
                                            output_face_blendshapes=True,
                                            result_callback=save_result)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = image

        if DETECTION_RESULT:
            # Draw landmarks.
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for
                    landmark in
                    face_landmarks
                ])
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

        # Expand the right side frame to show the blendshapes.
        current_frame = cv2.copyMakeBorder(current_frame, 0, 0, 0, label_padding_width, cv2.BORDER_CONSTANT, None, label_background_color)

        if DETECTION_RESULT:
          # Define parameters for the bars and text
          legend_x = current_frame.shape[1] - label_padding_width + 20  # Starting X-coordinate (20 as a margin)
          legend_y = 20  # Starting Y-coordinate
          bar_max_width = label_padding_width - 40  # Max width of the bar with some margin
          bar_height = 10  # Height of the bar
          gap_between_bars = 10  # Gap between two bars
          text_gap = 25  # Gap between the end of the text and the start of the bar

          face_blendshapes = DETECTION_RESULT.face_blendshapes


          if face_blendshapes:       
            #Filtra os blendshapes de 11 a 19, onde estao os blendshapes "eyeLook..."      
            for collumn in range(11,19):      
                category_name = face_blendshapes[0][collumn].category_name
                score = round(face_blendshapes[0][collumn].score, 2)





                # Prepare text and get its width
                text = "{} ({:.2f})".format(category_name, score)

                (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

                # Display the blendshape name and score
                cv2.putText(current_frame, text, (legend_x, legend_y + (bar_height // 2) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 1, cv2.LINE_AA) 

                # Calculate bar width based on score
                bar_width = int(bar_max_width * score)

                # Draw the bar to the right of the text
                cv2.rectangle(current_frame,
                                (legend_x + text_width + text_gap, legend_y),
                                (legend_x + text_width + text_gap + bar_width, legend_y + bar_height),
                                (0, 255, 0),  # Green color
                                -1)  # Filled bar

                # Update the Y-coordinate for the next bar
                legend_y += (bar_height + gap_between_bars) + 40

        cv2.imshow('face_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of face landmarker model.',
        required=False,
        default='face_landmarker.task')
    parser.add_argument(
        '--numFaces',
        help='Max number of faces that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minFaceDetectionConfidence',
        help='The minimum confidence score for face detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minFacePresenceConfidence',
        help='The minimum confidence score of face presence score in the face '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the face tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)

main()
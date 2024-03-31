import argparse
import sys
import time
import keyboard

import vlc
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

    #TODO: TESTAR SE REMOVER A VARIÁVEL ABAIXO JUNTO COM AS LINHAS 70 A 72 CAUSAM PROBLEMA
    fps_avg_frame_count = 10


    #Variáveis globais para calibrar a webcam
    threshold_LEFT = 1
    threshold_RIGHT = 1

    threshold_UP = 2
    Confirm_UP = 0

    threshold_DOWN = 2
    Confirm_DOWN = 0



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

    #Configura o video VLC
    instance = vlc.Instance("--no-xlib")

    player = instance.media_player_new()  
    # Creating media object
    media = instance.media_new("bavarian_alps_wimbachklamm_360.mp4")
    # Setting media to the player
    player.set_media(media)



    Ponto_de_Vista = vlc.libvlc_video_new_viewpoint()
    player.play()


    # Continuously capture images from the camera and run inference
    while cap.isOpened():

        success, current_frame = cap.read()

        '''
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )
        '''

        rgb_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        if DETECTION_RESULT:
          
          face_blendshapes = DETECTION_RESULT.face_blendshapes

          #As vezes, é possível detectar um resultado sem detectar os blendshapes

          if face_blendshapes:

            if (face_blendshapes[0][17].score + face_blendshapes[0][18].score) > threshold_UP and (face_blendshapes[0][11].score + face_blendshapes[0][12].score) < Confirm_UP: 

                cv2.putText(current_frame, "U P", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA) 

                Ponto_de_Vista.contents.pitch = -1

                vlc.libvlc_video_update_viewpoint(player, Ponto_de_Vista, False) 

            
            elif (face_blendshapes[0][11].score + face_blendshapes[0][12].score) > threshold_DOWN and (face_blendshapes[0][17].score + face_blendshapes[0][18].score) < Confirm_DOWN : 

                cv2.putText(current_frame, "D O W N", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)  

                Ponto_de_Vista.contents.pitch = 1

                vlc.libvlc_video_update_viewpoint(player, Ponto_de_Vista, False) 
            
        
  
            if (face_blendshapes[0][14].score + face_blendshapes[0][15].score) > threshold_LEFT : 

                cv2.putText(current_frame, "L E F T", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  

                Ponto_de_Vista.contents.yaw = -1 #Roda +3 graus no eixo Z 

                vlc.libvlc_video_update_viewpoint(player, Ponto_de_Vista, False) 
                
            elif (face_blendshapes[0][13].score + face_blendshapes[0][16].score) > threshold_RIGHT :  

                cv2.putText(current_frame, "R I G H T", (400, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                Ponto_de_Vista.contents.yaw = 1 #Roda +3 graus no eixo Z 

                vlc.libvlc_video_update_viewpoint(player, Ponto_de_Vista, False) 
        

        Ponto_de_Vista.contents.pitch = 0
        Ponto_de_Vista.contents.yaw = 0
            

        if keyboard.is_pressed('left'):

            threshold_LEFT = (face_blendshapes[0][14].score + face_blendshapes[0][15].score)
            print('LEFT side calibrated')

        elif keyboard.is_pressed('right'):

            threshold_RIGHT = (face_blendshapes[0][13].score + face_blendshapes[0][16].score)
            print('RIGHT side calibrated')

        elif keyboard.is_pressed('down'):
            threshold_DOWN = (face_blendshapes[0][11].score + face_blendshapes[0][12].score)
            Confirm_DOWN = (face_blendshapes[0][17].score + face_blendshapes[0][18].score)
            print('DOWN side calibrated')

        elif keyboard.is_pressed('up'):
            threshold_UP = (face_blendshapes[0][17].score + face_blendshapes[0][18].score)
            Confirm_UP = (face_blendshapes[0][11].score + face_blendshapes[0][12].score)
            print('UP side calibrated')

        cv2.imshow('Webcam', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break


    player.stop()
    player.release()
    instance.release()

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
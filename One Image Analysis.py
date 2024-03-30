import cv2
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import matplotlib.pyplot as plt

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

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

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# ----------------------------------------------CARREGAR A IMAGEM AQUI-------------------------------------------------------
image = mp.Image.create_from_file("Dan.jpg")
#-----------------------------------------------------------------------------------------------------------------------------


# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

frame_scores = np.array([blendshape.score for blendshape in detection_result.face_blendshapes[0]])
blendshape_names = [blendshape.category_name for blendshape in detection_result.face_blendshapes[0]]

plt.figure(figsize=(10, 6))
plt.bar(blendshape_names, frame_scores, color='skyblue', edgecolor='black')
plt.xlabel('Blendshape')
plt.ylabel('Valor do Blendshap')
plt.title('Blendshape x Valor do Blendshape')
plt.xticks(rotation=90)  # Rotating the x-axis labels by 45 degrees
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.show()


annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Imagem Analisada", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


# Fecha tudo
cv2.waitKey(0) 
cv2.destroyAllWindows() 

print(detection_result.face_blendshapes[9])
print(detection_result.face_blendshapes[10])
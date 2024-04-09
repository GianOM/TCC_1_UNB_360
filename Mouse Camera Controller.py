import vlc
import time
import cv2
import numpy as np

import subprocess #Usado para rodar a linha de comando

def eqruirect2persp(img, FOV, THETA, PHI, Height_Output, Widht_Output):

    # THETA is left/right angle, PHI is up/down angle, both in degree
    equ_h, equ_w = img.shape[:2]


    #Calcula centro da imagem
    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    #Campo de visão Horizontal e Vertical(em graus), sendo o vertical derivado do aspect ratio
    wFOV = FOV
    hFOV = float(Height_Output) / Widht_Output * wFOV

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))


    x_map = np.ones([Height_Output, Widht_Output], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len,Widht_Output), [Height_Output,1])
    z_map = -np.tile(np.linspace(-h_len, h_len,Height_Output), [Widht_Output,1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
    
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([Height_Output * Widht_Output, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

    lon = lon.reshape([Height_Output, Widht_Output]) / np.pi * 180
    lat = -lat.reshape([Height_Output, Widht_Output]) / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90  * equ_cy + equ_cy

        
    persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return persp


def draw_rectangle_on_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow('Frame with Rectangle', cv2.WINDOW_NORMAL)

    cv2.setWindowProperty('Frame with Rectangle', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.setWindowProperty('Frame with Rectangle', cv2.WND_PROP_TOPMOST, 1)#Configura a janela para aparecer assim a outra janela é fechada

    frame_delay = int(1000 / metadata.get('frame_rate'))

    idx_gaze_Map = 0

    #Se um Campo de Visao de 360 graus está para 1920 pixels
    #Entao, um campo de visao de X graus está para Y = (X * 1920)/360 PIXELS
    FOV_pxls_X = int( (Campo_de_Visao * 1920) / 360)

    #FOV_Y = np.radians(2 * np.arctan(1 / Campo_de_Visao))

    #As projeçoes equiretangulares não possuem a altura 1080. Para isto, verificou empiricamente a necessidade de corrigir a partir deste erro
    #TO DO: Justificar melhor a variável abaixo
    Height_Error =  int((metadata.get('Height') - 1080) / 2)
        
    FOV_pxls_Y = int(1.1 * FOV_pxls_X * metadata.get('Height') / metadata.get('Width'))

    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()
        #Atualmente o vídeo só roda até certo tempo. No futuro, se o certo tempo for maior do que a duração do vídeo
        #                                                                                           teremos problemas

        '''
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        '''

        #Por Fim, somamos e subtraimos o CTU para efeito de padding

        x_0, y_0 = (Gaze_Map[0][idx_gaze_Map] - FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map] + Height_Error - FOV_pxls_Y // 2)  
        x_1, y_1 = (Gaze_Map[0][idx_gaze_Map] + FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map] + Height_Error + FOV_pxls_Y // 2) 

        # cv2.rectangle(image, start_point,
        #                        end_point,
        #                        color, thickness)

        cv2.rectangle(frame, (x_0, y_0),
                             (x_1, y_1),
                             (0, 0, 255), 4)
        
        #Retangulos extras para o caso de sair da tela

        if x_0 < 0:
            #O retangulo sai da tela pra esquerda
            x_0 = metadata.get('Width') + x_0
            x_1 = x_0 + FOV_pxls_X
            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if y_0 < 0:

            #O retangulo sai da tela pra cima
            y_0 = metadata.get('Height') + y_0
            y_1 = y_0 + FOV_pxls_Y
            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if x_1 > metadata.get('Width'):

            #O retangulo sai da tela pra direita
            x_1 = x_1 - metadata.get('Width')
            x_0 = x_1 - FOV_pxls_X

            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if y_1 > metadata.get('Height'):

            #O retangulo sai da tela pra baixo        
            y_1 = y_1 - metadata.get('Height')
            y_0 = y_1 - FOV_pxls_Y

            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)
        

        # Display the resulting frame
        cv2.imshow('Frame with Rectangle', frame)

        # Break the loop on pressing 'q'. Sets the framerate to 30fps
        if (cv2.waitKey(frame_delay) & 0xFF == ord('q')) or ( idx_gaze_Map + 1 == Numero_de_Frames):
            break

        idx_gaze_Map = idx_gaze_Map + 1

    cap.release()
    cv2.destroyAllWindows()


def map_value(value, in_min, in_max, out_min, out_max):
    # Map the value from the input range to the output range
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def Automatic_Video(video_path, tempo):

    # Creating VLC instance
    instance = vlc.Instance("--input-repeat=-1", "--no-xlib")
    
    # Creating VLC media player object
    player = instance.media_player_new()
    
    # Creating media object
    media = instance.media_new(video_path)

    # Setting media to the player
    player.set_media(media)
    
    VLC_Ponto_de_Vista = vlc.libvlc_video_new_viewpoint()

    player.set_rate(1)
    player.set_fullscreen(True)

    player.play()

    # Cursor position ranges
    x_min, x_max = 0, 1919
    y_min, y_max = 0, 1079

    # Desired output ranges
    yaw_min, yaw_max = -180, 180
    pitch_min, pitch_max = -90, 90

    VLC_Ponto_de_Vista.contents.field_of_view = Campo_de_Visao
    
    #Grave o movimento do mouse
    for i in range(tempo):   

        x, y = Gaze_Map[0][i], Gaze_Map[1][i] = player.video_get_cursor()

        #DEBUG TOOL:
        #x, y = vector[0][i], vector[1][i] = metadata.get('resolution')[0], metadata.get('resolution')[1]

        VLC_Ponto_de_Vista.contents.yaw = map_value(x, x_min, x_max, yaw_min, yaw_max) # -> Equivale a rodar a cabeça da esquerda para direita
        VLC_Ponto_de_Vista.contents.pitch =  map_value(y, y_min, y_max, pitch_min, pitch_max) # -> Equivale a rodar a cabeça para cima e para baixo

        vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)
        #vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True) -> Se o ultimo parametro for true, ele troca o viewpoint. Se for false, ele soma ou
        #                                                                       subtrai o viewpoint

        time.sleep(1/metadata.get('frame_rate'))

    player.release()
    instance.release()


def draw_grid(img, density):

    height, width = img.shape[:2]

    vertical_interval = width // density
    horizontal_interval = height // density

    color = (0, 0, 255)  # Red in BGR format
    thickness = 8  # Adjust thickness as needed

    for i in range(1, 17):

        start_point = (0, i * horizontal_interval)
        end_point = (width, i * horizontal_interval)
        cv2.line(img, start_point, end_point, color, thickness)

    for i in range(1, 17):

        start_point = (i * vertical_interval, 0)
        end_point = (i * vertical_interval, height)
        cv2.line(img, start_point, end_point, (255, 0, 0), thickness)

    
    return img


def set_Video_360_Metada(media):

    subprocess.run(['exiftool', '-XMP-GSpherical:Spherical="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:Stitched="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:StitchingSoftware="Spherical Metadata Tool', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:ProjectionType="equirectangular', media])


def get_video_metadata(video_path):

    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'default=nw=1:nk=1', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    
    metadata = {}
    
    output_lines = result.stdout.strip().split('\n')
    
    metadata['Width'] = int(output_lines[0])
    metadata['Height'] = int(output_lines[1])
    
    # Convert framerate to integer
    metadata['frame_rate'] = eval(output_lines[2])
    
    # Convert duration to float
    metadata['duration'] = float(output_lines[3])
    
    return metadata



#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------

input_video = '18_360_Carnival_of_Venice_Italy_4k_video.mp4'

metadata = get_video_metadata(input_video)

set_Video_360_Metada(input_video)

#Numero_de_Frames = Segundos x Frame/Segundo
Numero_de_Frames = int(metadata.get('duration') * metadata.get('frame_rate'))

gaze_Map_X = [0 for _ in range(Numero_de_Frames)]
gaze_Map_Y = [0 for _ in range(Numero_de_Frames)]

Gaze_Map = [gaze_Map_X, gaze_Map_Y]

Campo_de_Visao = 90

#Coding Tree Unit
CTU = 0

Automatic_Video(input_video, Numero_de_Frames)

draw_rectangle_on_video(input_video)


'''

cap = cv2.VideoCapture(input_video)

# Read and display video frames in a loop
while True:
    # Read a new frame
    ret, frame = cap.read()

    frame = eqruirect2persp(frame, 100, 0, 0, 960, 1920)

    cv2.imshow('Video Playback', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()


#Pontos do quadrado no sentido horário

def create_perspective_projection_matrix(f, sar, zNear, zFar):
    """
    Create a 4x4 perspective projection matrix.

    Parameters:
    - f: Focal length.
    - sar: Aspect ratio (Screen Aspect Ratio), typically width / height of the image.
    - zNear: Distance to the near clipping plane.
    - zFar: Distance to the far clipping plane.

    Returns:
    - A 4x4 NumPy array representing the perspective projection matrix.
    
    return np.array
    """
    return np.array([
                      [ f/sar,  0,                   0,                      0],
                      [  0,     f,                   0,                      0],
                      [  0,     0,  ((zNear + zFar) / (zNear - zFar)),      -1],
                      [  0,     0,  ((2 * zNear * zFar) / (zNear - zFar)),   0]]) 



([
                    [1, 0,  0, 0],
                    [0, 1,  0, 0],
                    [0, 0,  1, 0],
                    [0, 0,  0, 1]])

([
                    [0.9659, 0.2588,   0, 0],
                    [-0.2588, 0.9659, -0, 0],
                    [0,         0,     1, 0],
                    [0,         0,    0, 1]])

([
                    [0.866, 0.5,   0, 0],
                    [-0.5, 0.866, 0, 0],
                    [0,         0,     1, 0],
                    [0,         0,    0, 1]])


([
                      [ f/sar,  0,                   0,                      0],
                      [  0,     f,                   0,                      0],
                      [  0,     0,  ((zNear + zFar) / (zNear - zFar)),      -1],
                      [  0,     0,  ((2 * zNear * zFar) / (zNear - zFar)),   0]])                    


def center_Image_at_origin(image):
    
    # Find the center of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    
    # Calculate the shift required to move the image center to the origin
    # Since we're moving the center to the origin, we need to shift it by half its dimensions in the negative direction
    tx, ty = -center_x, -center_y
    
    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply the translation
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    return translated_image


def restore_image_center(translated_image):
    # Assuming translated_image is the image with its center at the origin (0,0)
    
    # Calculate the shift required to move the image back to its original center
    # This is simply half the dimensions in the positive direction
    tx, ty = translated_image.shape[1] // 2, translated_image.shape[0] // 2
    
    # Create the reverse translation matrix
    reverse_translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply the reverse translation
    restored_image = cv2.warpAffine(translated_image, reverse_translation_matrix, (translated_image.shape[1], translated_image.shape[0]))
    
    return restored_image



def prepare_pixel_coordinates(width, height):
    """
    Prepare the pixel coordinates for an image of given width and height, treating each pixel as a point in 3D space (x, y, 0).
    
    Parameters:
    - width: Width of the image.
    - height: Height of the image.
    
    Returns:
    - A 3D NumPy array of shape (height, width, 4), where each element is a 4D point (x, y, z, w) with z = 0 and w=1.
    """
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)
    # Create a grid of points with z=0 and w=1 (homogeneous coordinate)
    points_3d = np.stack((xv, yv, np.zeros_like(xv), np.ones_like(xv)), axis=-1)
    return points_3d

def apply_perspective_transformation(points_3d, perspective_matrix):
    """
    Apply a perspective transformation to a set of 3D points using a given perspective projection matrix.
    
    Parameters:
    - points_3d: A 3D NumPy array of shape (height, width, 4), where each element is a 4D point (x, y, z, w).
    - perspective_matrix: The 4x4 perspective projection matrix.
    
    Returns:
    - A 3D NumPy array of transformed points, in homogeneous coordinates.
    """
    # Reshape points_3d for matrix multiplication: from (height, width, 4) to (height * width, 4)
    reshaped_points = points_3d.reshape(-1, 4)
    # Apply the perspective transformation
    transformed_points = np.dot(reshaped_points, perspective_matrix.T)  # Transpose matrix for correct multiplication
    # Reshape back to the original shape, but points are now in homogeneous coordinates
    return transformed_points.reshape(points_3d.shape)

# Example usage of the matrix creation function (parameters are placeholders and should be set appropriately)
f = np.pi / 2  # Focal length
sar = 2  # Screen aspect ratio, for a 1920x1080 image
zNear = 0.1  # Near clipping plane
zFar = 1000  # Far clipping plane

# Create the perspective projection matrix
perspective_matrix = create_perspective_projection_matrix(f, sar, zNear, zFar)

# Assuming the image dimensions are known (1920x960 in this case)
width, height = 1920, 960

# Prepare the pixel coordinates
pixel_coordinates = prepare_pixel_coordinates(width, height)

# Apply the perspective transformation
# Note: 'perspective_matrix' should be defined as shown in the previous example
transformed_points = apply_perspective_transformation(pixel_coordinates, perspective_matrix)

# The next step would involve interpolating these transformed points back onto a 2D image grid.

# Normalize the transformed points
# Assume 'transformed_points' is the output from the previous step

#ALGO ESTÁ DANDO ERRADO AQUI
normalized_points = transformed_points[:, :, :2] / transformed_points[:, :, 3:4]

#normalized_points = transformed_points[:, :, :2] / 1

# Prepare the maps for remapping
# Note: OpenCV's remap function expects the map to be in floating-point format
map_x = normalized_points[:, :, 0].astype(np.float32)
map_y = normalized_points[:, :, 1].astype(np.float32)

# Load the original image
# Replace 'image_path' with the path to your image
original_image = cv2.imread('Teste.png')
#original_image = center_Image_at_origin(original_image)

# Perform the remapping to get the transformed image
# This maps the original image pixels to their new locations based on the transformation
transformed_image = cv2.remap(original_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

#transformed_image = restore_image_center(transformed_image)

cv2.imshow('transformed_image.png', transformed_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
'''
import vlc
import time
import pyautogui
import cv2

import numpy as np

def eqruirect2persp(img, FOV, THETA, PHI, Height_Output, Widht_Output):

    # THETA is left/right angle, PHI is up/down angle, both in degree
    equ_h, equ_w = img.shape[:2]


    #Calcula centro da imagem
    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    #Campo de visão Horizontal e Vertical(em graus), sendo o vertical derivado do aspect ratio
    wFOV = FOV
    hFOV = float(Height_Output) / Widht_Output * wFOV


    #Centro da projeção retilínea
    c_x = (Widht_Output) / 2.0
    c_y = (Height_Output) / 2.0

    w_len = 2 * np.tan(np.radians(wFOV / 2.0))
    w_interval = w_len / (Widht_Output)

    h_len = 2 * np.tan(np.radians(hFOV / 2.0))
    h_interval = h_len / (Height_Output)

    #what the fuck is happening here????
    x_map = np.zeros([Height_Output, Widht_Output], np.float32) + 1 #Cria uma matriz de 1's do tamanho do widht da saida
    y_map = np.tile((np.arange(0, Widht_Output) - c_x) * w_interval, [Height_Output, 1])
    z_map = -np.tile((np.arange(0, Height_Output) - c_y) * h_interval, [Widht_Output, 1]).T
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

    xyz = np.zeros([Height_Output, Widht_Output, 3], float)

    xyz[:, :, 0] = (x_map / D)[:, :]
    xyz[:, :, 1] = (y_map / D)[:, :]
    xyz[:, :, 2] = (z_map / D)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([Height_Output * Widht_Output, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T


    lat = np.arcsin(xyz[:, 2] / 1)
    lon = np.zeros([Height_Output * Widht_Output], float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([Height_Output, Widht_Output]) / np.pi * 180
    lat = -lat.reshape([Height_Output, Widht_Output]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy


    return cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)



# Function to draw a rectangle on each frame of a video
def draw_rectangle_on_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow('Frame with Rectangle', cv2.WINDOW_NORMAL)

    cv2.setWindowProperty('Frame with Rectangle', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_delay = int(1000 / 30)

    idx_gaze_Map = 0

    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()
        #Atualmente o vídeo só rota até certo tempo. No futuro, se o certo tempo for maior do que a duração do vídeo
        #                                                                                           teremos problemas


        # cv2.rectangle(image, start_point, end_point, color, thickness)

        x_0, y_0 = (720 + vector[0][idx_gaze_Map] - 960, 335 + vector[1][idx_gaze_Map] - 480)  
        x_1, y_1 = (x_0 + 480, y_0 + 240) 


        cv2.rectangle(frame, (x_0, y_0),
                             (x_1, y_1),
                             (0, 0, 255), 4)
        
        #Retangulo fora da tela
        if x_0 < 0:
            x_0 = 1920 + x_0
            x_1 = x_0 + 480
            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if y_0 < 0:
            y_0 = 1080 + y_0
            y_1 = y_0 + 240
            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if x_1 > 1920:

            x_1 = x_1 - 1920
            x_0 = x_1 - 480

            cv2.rectangle(frame, (x_0, y_0),
                                 (x_1, y_1),
                                 (0, 0, 255), 4)

        if y_1 > 1080:
                        
            y_1 = y_1 - 1080
            y_0 = y_1 - 240

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

    VLC_Ponto_de_Vista.contents.field_of_view = 90
    
    #vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True) -> Se o ultimo parametro for true, ele troca o viewpoint. Se for false, ele soma ou
    #                                                             subtrai o viewpoint


    vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)
    
    #Grave o movimento do mouse
    for i in range(tempo):   

        x, y = vector[0][i], vector[1][i] = pyautogui.position()

        VLC_Ponto_de_Vista.contents.yaw = map_value(x, x_min, x_max, yaw_min, yaw_max) # -> Equivale a rodar a cabeça da esquerda para direita
        VLC_Ponto_de_Vista.contents.pitch =  map_value(y, y_min, y_max, pitch_min, pitch_max) # -> Equivale a rodar a cabeça para cima e para baixo

        vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)

        time.sleep(1/60)

    player.stop()
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


#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------

#Levando em consideração que esta mídia possui 30fps


Numero_de_Frames = 15 * 30 #Segundos x Frame/Segundo

gaze_Map_X = [0 for _ in range(Numero_de_Frames)]
gaze_Map_Y = [0 for _ in range(Numero_de_Frames)]

#print(gaze_Map_X.len())

vector = [gaze_Map_X, gaze_Map_Y]

video_path = "bavarian_alps_wimbachklamm_360.mp4"


Automatic_Video(video_path, Numero_de_Frames)

draw_rectangle_on_video('bavarian_alps_wimbachklamm_Equiretangular.mp4')


'''
cap = cv2.VideoCapture("bavarian_alps_wimbachklamm_Equiretangular.mp4")


yawn_position = 0
pitch_position = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = draw_grid(frame, 16)

    #                       img,  FOV, THETA, PHI, WIDHT_OUTPUT, HEIGHT_OUTPUT
    frame = eqruirect2persp(frame, 90, yawn_position, pitch_position, 960, 1920)
    cv2.imshow('Frame', frame)

    #yawn_position = yawn_position + 1

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows() 



# Draw vertical lines
Imagem_Equiretangular_Panoramica = cv2.imread("Teste.png")
Imagem_com_Grids = draw_grid(Imagem_Equiretangular_Panoramica)

# Display the image with lines
cv2.imshow('Image with Red Lines', Imagem_Equiretangular_Panoramica)


teste2 = eqruirect2persp(Imagem_com_Grids, 114, 0, 0, 720, 1280)


cv2.imshow('Image Window', teste2)

cv2.waitKey(0)
cv2.destroyAllWindows()

#draw_rectangle_on_video("bavarian_alps_wimbachklamm_Equiretangular.mp4")
'''

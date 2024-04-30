import vlc
import time #Usado para controlar o frame rate do vídeo. TO DO: REMOVE THIS LIBRARY
import cv2
import numpy as np

import os#Usado para ler e escrever os arquivos nas pastas corretas

from threading import Thread # library for implementing multi-threaded processing
from queue import Queue

import cProfile
import subprocess #Usado para rodar a linha de comando

def Calculate_Atlas_per_Frame(M, N, index_Atlas, Atlas, X_Center, Y_Center):

    #A ideia desta função é apenas calcular o atlas para certo frame
    #Calcula a distância do centro do Gaze_Map para o centro de cada quadrado  
    for j in range(M):
        for k in range(N):
            center_x = (j * piece_width) + piece_width // 2
            center_y = (k * piece_height) + piece_height // 2

            dist_x = min(abs(X_Center - center_x), metadata.get('Width') - abs(X_Center - center_x))
            #Não existe screen border wrapping no eixo Y

            Atlas[j, k, index_Atlas] += np.sqrt(dist_x**2  + (Y_Center - center_y)**2)       


def Calculate_Heat_Map(video_path, Atlas):

    #Se um Campo de Visao de 360 graus está para 1920 pixels da minha tela,
    #entao, um campo de visao de X graus está para Y = (X * 1920)/360 PIXELS

    #FOV_pxls_X = int( (Campo_de_Visao * 1920) / 360)
    #FOV_pxls_Y = int(FOV_pxls_X * metadata.get('Height') / metadata.get('Width'))

    #FOV_Y = np.radians(2 * np.arctan(1 / Campo_de_Visao)) -> O VLC calcula o FOV vertical em pixels a partir desta formula
 

    idx_Atlas = 0

    # Load the video
    captura_de_video = Video_Capturing(video_path)
    
    for frame_number_i in range(Numero_de_Frames):
        #As projeçoes equiretangulares não possuem a altura 1080, o que significa que o vídeo possui uma
        #resolução diferente da tela. Por isto, devemos inserir a resolução DA TELA QUE ESTÁ SENDO OBSERVADA
        #frame = cv2.resize(frame, (1920, 1080))
        if ((frame_number_i % 5) == 0):
            #Faz o cv2 pegar um a cada 5 frames para reduzir drasticamente o tempo de execução do programa
            frame = captura_de_video.grab_frame()
        # (x_0, y_0)---------------------------------------------------
        #      |                                                       |  
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      |                                                       |
        #      ----------------------------------------------------(x_1, y_1)
        

        #x_0, y_0 = (Gaze_Map[0][idx_gaze_Map] - FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map]  - FOV_pxls_Y // 2)  
        #x_1, y_1 = (Gaze_Map[0][idx_gaze_Map] + FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map]  + FOV_pxls_Y // 2) 
        #Calculate_Atlas_per_Frame( M, N, idx_Atlas, Atlas, (x_0 + x_1)//2, (y_0 + y_1)//2)

        Calculate_Atlas_per_Frame(M, N, idx_Atlas, Atlas, Gaze_Map[0][frame_number_i], Gaze_Map[1][frame_number_i])

        if (frame_number_i + 1 == ((Numero_de_Frames // K) * (idx_Atlas + 1))):

            success, frame = captura_de_video.retrieve_frame()

            Distancia_Maxima = np.max(Atlas[:, :, idx_Atlas])
            Atlas[:, :, idx_Atlas] = Atlas[:, :, idx_Atlas] / Distancia_Maxima
            Atlas[:, :, idx_Atlas] = np.round(Atlas[:, :, idx_Atlas], decimals=3)

            Draw_Atlas(M,N,idx_Atlas, Gaze_Atlas, frame)
            Create_ROI_Lookup_file(M,N,idx_Atlas, Gaze_Atlas)

            idx_Atlas += 1

    captura_de_video.stop()
    return Atlas



def Draw_Atlas(M, N, index_Atlas, Atlas, Drawing_frame):
    #Recebe um frame do array Lista_de_Frames_para_desenhar e repassa este frame para ser desenhado
    frame = Drawing_frame

    for i in range(1, M):
        cv2.line(frame, (i * piece_width, 0), (i * piece_width, metadata.get('Height')), (0, 0, 0), 4)
    for i in range(1, N):
        cv2.line(frame, (0, i * piece_height), (metadata.get('Width'), i * piece_height), (0, 0, 0), 4)
        
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2

    for j in range(M):
        for k in range(N):
            #Determina o centro de um pedaço
            center_x = j * piece_width + piece_width // 2
            center_y = k * piece_height + piece_height // 2

            #              cv2.getTextSize(text,                               font,         font_scale,   thickness)
            text_size, _ = cv2.getTextSize(str(Atlas[j, k, index_Atlas]), font, font_scale, font_thickness)
            text_width, text_height = text_size

            cv2.putText(frame, str(Atlas[j, k, index_Atlas]), (center_x - text_width // 2, center_y - text_height // 2),
                                 font, font_scale, (0, 0, (1-Atlas[j, k, index_Atlas])*255), font_thickness, cv2.LINE_AA)
            
    #Se escrevermos o arquivo em .jpg -> 41ms pra escrever por frame
    #Já se escrevermos em .bmp -> 6ms pra escrever por frame
    file_path = os.path.join(Intermediate_Files_Folder, f'Atlas {(Numero_de_Frames // K) * (index_Atlas)} to {(Numero_de_Frames // K) * (index_Atlas + 1)} frames.bmp')
    #A imagem ta saindo com a resolução 1920x1440, que é a resolução original. Por isto, damos um resize nela
    #cv2.imwrite(file_path, cv2.resize(frame, (1920, 1080)))
    cv2.imwrite(file_path, frame)
    
    return frame


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

    fps = metadata.get('frame_rate')

    # Pixels minimos e máximos do vídeo
    x_min, x_max = 0, int(metadata.get('Width')) - 1
    y_min, y_max = 0, int(metadata.get('Height')) - 1

    # Desired output ranges
    yaw_min, yaw_max = -180, 180
    pitch_min, pitch_max = -90, 90

    VLC_Ponto_de_Vista.contents.field_of_view = Campo_de_Visao
    
    #Grave o movimento do mouse
    for i in range(tempo):   

        x, y = Gaze_Map[0][i], Gaze_Map[1][i] = player.video_get_cursor()

        #DEBUG TOOL:
        #x, y = Gaze_Map[0][i], Gaze_Map[1][i] = 4096, 2048

        VLC_Ponto_de_Vista.contents.yaw = map_value(x, x_min, x_max, yaw_min, yaw_max) # -> Equivale a rodar a cabeça da esquerda para direita
        VLC_Ponto_de_Vista.contents.pitch =  map_value(y, y_min, y_max, pitch_min, pitch_max) # -> Equivale a rodar a cabeça para cima e para baixo


        vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)
        #vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True) -> Se o ultimo parametro for true, ele troca o viewpoint. Se for false, ele soma ou
        #                                                                       subtrai o viewpoint

        time.sleep(1/fps)

    player.release()
    instance.release()




def set_Video_360_Metada(media):
    #Insere os metadados para o video player entender que se trata de um video 360
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



def Create_ROI_Lookup_file(M,N,i, Atlas):

    ROI_Text = []

    for x in range(M):
        for y in range(N):
            center_x = x * piece_width
            center_y = y * piece_height

            qp_offset = (2* Atlas[x, y, i]) - 1
            qp_offset = np.round(qp_offset, decimals=5)

            ROI_Text += "addroi=" + str(center_x) + ":" + str(center_y) + ":" + str(piece_width) + ":" + str(piece_height) + ":" + str(qp_offset) + ", "

    ROI_Text = ROI_Text[:-2]#Remove os dois ultimos caracteres ", " para evitar erros
    String_ROI_Text = ''.join(map(str,ROI_Text))

    file_path = os.path.join(ROI_Lookup_Files_Folder, f'ROI_LOOKUP_TEXT_{i}.txt')

    with open(file_path, 'w') as file:
        file.write(String_ROI_Text)




#---------------------------------------------------------------------Classes-------------------------------------------------------------------------------------------------
class Video_Capturing():
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)        

    def grab_frame(self):
        self.video_capture.grab()

    def retrieve_frame(self):
        return self.video_capture.retrieve()
    
    def stop(self):
        self.video_capture.release()
        cv2.destroyAllWindows()






#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------

input_video = "SiyuanGate.mp4"
set_Video_360_Metada(input_video)

metadata = get_video_metadata(input_video)


Diretorio_Atual = os.getcwd()
Intermediate_Files_Folder = os.path.join(Diretorio_Atual, 'Intermediate Files')
ROI_Lookup_Files_Folder = os.path.join(Diretorio_Atual, 'ROI Lookup Files')


Numero_de_Frames = int(metadata.get('duration') * metadata.get('frame_rate'))
#Numero_de_Frames = 25*12


#A variável Gaze Map captura a cada frame o centro da tela para onde a pessoa está olhando
gaze_Map_X = [0 for _ in range(Numero_de_Frames)]
gaze_Map_Y = [0 for _ in range(Numero_de_Frames)]

Gaze_Map = [gaze_Map_X, gaze_Map_Y]

# Divide o vídeo em MxN regioes e cria uma matrix MxN
# "K" é o número de "Gaze Atlas" que queremos criar. Podemos ter um Gaze_Atlas para 0 a 30s e outro para 30 a 60s
(M, N, K) = 8,8,6
Gaze_Atlas = np.zeros((M, N, K))

#Tamanho do pedaço da Região de Interesse
piece_width = metadata.get('Width') // M
piece_height = metadata.get('Height') // N


#Define o campo de visão a ser usado no vídeo
Campo_de_Visao = 90

Automatic_Video(input_video, Numero_de_Frames)

#cProfile.run('Gaze_Atlas = Calculate_Heat_Map(input_video, Gaze_Atlas)')

Gaze_Atlas = Calculate_Heat_Map(input_video, Gaze_Atlas)
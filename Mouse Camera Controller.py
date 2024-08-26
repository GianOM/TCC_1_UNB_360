import vlc
import time #Usado para controlar o frame rate do vídeo. TO DO: REMOVE THIS LIBRARY
import cv2
import numpy as np

import math

import os#Usado para ler e escrever os arquivos nas pastas corretas

import cProfile#Usado para profiling do código e debuggar bottlenecks
import subprocess #Usado para rodar a linha de comando


def Contrast_Threshold(Number_of_X_Tiles, Number_of_Y_Tiles):

    #Descobre quantos graus theta e phi tem de uma tile para outra, considerando que o video tenha 360º na horizontal e 180º na vertical
    Horizontal_Tile_Degrees = (360*(piece_width//2))//metadata.get('Width')
    Vertical_Tile_Degrees = (180*(piece_height//2))//metadata.get('Height')


    #Em seguida, esta distancia em graus é usada para calcular a excentricidade
    Excentricity = Horizontal_Tile_Degrees*Number_of_X_Tiles + Vertical_Tile_Degrees*Number_of_Y_Tiles

    Excentricity = 1/64 * (math.exp(0.106*(1 + Excentricity/2.3)))

    #A constante 0.017372 representa excentricidade igual a 0, ou seja, o individuo olhou diretamente para aquela tile
    Excentricity = 0.017372 / Excentricity
    

    return Excentricity
    


def Calculate_Atlas_per_Frame(index_Atlas, Atlas, X_Center, Y_Center):

    #Descubro a tile que estou olhando
    Tile_X,Tile_Y = X_Center//piece_width, Y_Center//piece_height

    #Para a tile que eu estou olhando, ela ganha +1
    Atlas[Tile_X, Tile_Y, index_Atlas] += Contrast_Threshold(0,0)


    #Para as tiles adjacentes, ela ganha:
    # CT(Exc) = (1/64)*Exp[0.106*(1 + Exc/2.3)]
    # spie1998.pdf, equation 1

    for i in range(1,2):

        # Horizontal wrap-around handling
        left_x = (Tile_X - i) % M
        right_x = (Tile_X + i) % M
        
        # Vertical
        up_y = (Tile_Y - i) % N
        down_y = (Tile_Y + i) % N


        Atlas[Tile_X, up_y, index_Atlas] += Contrast_Threshold(0,i)#Up
        Atlas[left_x, up_y, index_Atlas] += Contrast_Threshold(i,i)#Up-Left
        Atlas[right_x, up_y, index_Atlas] += Contrast_Threshold(i,i)#Up-Right

        Atlas[left_x, Tile_Y, index_Atlas] += Contrast_Threshold(i,0)#Left
        Atlas[right_x, Tile_Y, index_Atlas] += Contrast_Threshold(i,0)#Right

        Atlas[Tile_X, down_y, index_Atlas] += Contrast_Threshold(0,i)#Bottom
        Atlas[right_x , down_y, index_Atlas] += Contrast_Threshold(i,i)#Bottom-Right
        Atlas[left_x, down_y, index_Atlas] += Contrast_Threshold(i,i)#Bottom-Left



     


def Calculate_Heat_Map(video_path, Atlas):

    #Se um Campo de Visao de 360 graus está para 1920 pixels da minha tela,
    #entao, um campo de visao de X graus está para Y = (X * 1920)/360 PIXELS

    idx_Atlas = 0

    # Load the video
    captura_de_video = Video_Capturing(video_path)
    
    for frame_number_i in range(Numero_de_Frames):
        #As projeçoes equiretangulares não possuem a altura 1080, o que significa que o vídeo possui uma
        #resolução diferente da tela. Por isto, devemos inserir a resolução DA TELA QUE ESTÁ SENDO OBSERVADA
        #frame = cv2.resize(frame, (1920, 1080))
        if ((frame_number_i % 12) == 0):
            #Faz o cv2 pegar um a cada 12 frames para reduzir drasticamente o tempo de execução do programa
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
        

        Calculate_Atlas_per_Frame(idx_Atlas, Atlas, Gaze_Map[0][frame_number_i], Gaze_Map[1][frame_number_i])

        if (frame_number_i + 1 >= ((Numero_de_Frames / K) * (idx_Atlas + 1))):

            success, frame = captura_de_video.retrieve_frame()

            Normalize_Atlas(Atlas, idx_Atlas)

            Draw_Atlas(M,N,idx_Atlas, Atlas, frame)

            #Create_ROI_Lookup_file(M,N,idx_Atlas, Atlas)

            Create_Kvazaar_ROI(M,N,idx_Atlas, Atlas)

            idx_Atlas += 1

    captura_de_video.stop()

    return Atlas

def Normalize_Atlas(Atlas, index_Atlas):

    Distancia_Maxima = np.max(Atlas[:, :, index_Atlas])
    Atlas[:, :, index_Atlas] = Atlas[:, :, index_Atlas] / Distancia_Maxima

    #Comentar a linha de baixo se estiver o usando o modo automático
    #Atlas[:, :, index_Atlas] = 1 - Atlas[:, :, index_Atlas]

    Atlas[:, :, index_Atlas] = np.round(Atlas[:, :, index_Atlas], decimals=3)




def Draw_Atlas(M, N, index_Atlas, Atlas, Drawing_frame):
    #Recebe um frame do array Lista_de_Frames_para_desenhar e repassa este frame para ser desenhado
    
    Drawing_frame = draw_grid(Drawing_frame)
        
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2

    for j in range(M):
        for k in range(N):
            #Determina o centro de um pedaço
            center_x = j * piece_width + piece_width // 2
            center_y = k * piece_height + piece_height // 2

            #              cv2.getTextSize(text,                               font,         font_scale,   thickness)
            text_size, _ = cv2.getTextSize(str(Atlas[j, k, index_Atlas]), font, font_scale, font_thickness)
            text_width, text_height = text_size

            cv2.putText(Drawing_frame, str(Atlas[j, k, index_Atlas]), (center_x - text_width // 2, center_y - text_height // 2),
                                 font, font_scale, (0, 0, (1-Atlas[j, k, index_Atlas])*255), font_thickness, cv2.LINE_AA)




    '''
    First argument is the center location (x,y)
    Next argument is axes lengths (major axis length, minor axis length).
    The angle of rotation of ellipse in anti-clockwise direction.
    startAngle and endAngle denotes the starting and ending of ellipse arc measured in clockwise direction from major axis( 0 and 360 gives the full ellipse)
    Thickness of the elipse. If negative, draw a filled elipse
    
    #Elipse de 60 graus
    cv2.ellipse(Drawing_frame,(120,90),(320,480),0,0,360,(33,33,33),8)

    # 1920 ------- 360º
    #   x  ------- 60º    -> x = 320 

    # 1440 ------- 180º
    #   y  ------- 60º    -> y = 480 

    cv2.ellipse(Drawing_frame,(120,90),(240,360),0,0,360,(42,42,42),8)#45 Graus Elipse
    cv2.ellipse(Drawing_frame,(120,90),(160,240),0,0,360,(70,70,70),8)#30 Graus Elipse
    cv2.ellipse(Drawing_frame,(120,90),(80,120),0,0,360,(148,148,148),8)#15 Graus Elipse
    '''
        
    #Se escrevermos o arquivo em .jpg -> 41ms pra escrever por frame
    #Já se escrevermos em .bmp -> 6ms pra escrever por frame

    file_path = os.path.join(Intermediate_Files_Folder, f'Atlas {(Numero_de_Frames // K) * (index_Atlas)} to {(Numero_de_Frames // K) * (index_Atlas + 1)} frames.bmp')
    #A imagem ta saindo com a resolução 1920x1440, que é a resolução original. Por isto, damos um resize nela
    #cv2.imwrite(file_path, cv2.resize(frame, (1920, 1080)))
    cv2.imwrite(file_path, Drawing_frame)
    
    return Drawing_frame


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

        Gaze_Map[0][i], Gaze_Map[1][i] = player.video_get_cursor()

        #DEBUG TOOL que olha pro centro da projeção 3D, ou seja, Yaw = 90 e Pitch = 45:
        #Gaze_Map[0][i], Gaze_Map[1][i] = int(metadata.get('Width'))/2, int(metadata.get('Height'))/2

        VLC_Ponto_de_Vista.contents.yaw = map_value(player.video_get_cursor()[0], x_min, x_max, yaw_min, yaw_max) # -> Equivale a rodar a cabeça da esquerda para direita
        VLC_Ponto_de_Vista.contents.pitch =  map_value(player.video_get_cursor()[1], y_min, y_max, pitch_min, pitch_max) # -> Equivale a rodar a cabeça para cima e para baixo


        vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)
        #Se o ultimo parametro for true, ele troca o viewpoint. Se for false, ele soma ou subtrai o viewpoint

        time.sleep(1/fps)

    player.release()
    instance.release()

def mouse_callback(event, x, y, flags, param):

    Marked_Image, Clean_Image, Final_K = param

    if event == cv2.EVENT_LBUTTONDOWN:

        Tile_X, Tile_Y = (x // piece_width), (y // piece_height)

        Gaze_Atlas[Tile_X, Tile_Y, Final_K] += 1

        Background_Image = Marked_Image.copy()

        cv2.rectangle(Background_Image, (Tile_X*piece_width, Tile_Y*piece_height), (Tile_X*piece_width + piece_width, Tile_Y*piece_height + piece_height), (0,0,255), -1)
        cv2.addWeighted(Background_Image, 0.25, Marked_Image, 0.75, 0, Marked_Image)

            


    elif event == cv2.EVENT_RBUTTONDOWN:

        Tile_X, Tile_Y = (x // piece_width), (y // piece_height)

        Gaze_Atlas[Tile_X, Tile_Y, Final_K] -= 1

        Another_Temporary_Image = Clean_Image.copy()

        for i in range(M):
            for j in range(N):
                z = int(Gaze_Atlas[i, j ,Final_K])
                if z > 0:
                    for k in range(z):
                            Background_Image = Another_Temporary_Image.copy()

                            cv2.rectangle(Background_Image, (i*piece_width, j*piece_height), (i*piece_width + piece_width, j*piece_height + piece_height), (0,0,255), -1)

                            cv2.addWeighted(Background_Image, 0.25, Another_Temporary_Image, 0.75, 0, Another_Temporary_Image)


         
        #Background, Alpha_Background, Foreground, Alpha Foregroun, Gamma, Destination image
        cv2.addWeighted(Another_Temporary_Image, 1, Marked_Image, 0, 0, Marked_Image)
        #APARENTEMENTE, ESTE É O UNICO MODO DE PARSE DATA USANDO MOUSE CALLBACKS

                    

def Select_Manual_ROI(video_path):
    

    captura_de_video = Video_Capturing(video_path)
    captura_de_video.grab_frame()

    success, frame = captura_de_video.retrieve_frame()

    frame = draw_grid(frame)

    frame_backup = frame.copy()

    K_iterator = 0 #Variável usada  para acessar diferentes Gaze_Atlases

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback, (frame, frame_backup, K_iterator))

    while(K_iterator < K):
        
        if (cv2.waitKey(5) & 0xFF) == ord('m'):

            Normalize_Atlas(Gaze_Atlas, K_iterator)

            Draw_Atlas(M,N,K_iterator,Gaze_Atlas,frame)

            #Create_ROI_Lookup_file(M,N,K_iterator, Gaze_Atlas)
            Create_Kvazaar_ROI(M,N,K_iterator, Gaze_Atlas)

            K_iterator += 1

            captura_de_video.set_frame(K_iterator * piece_frame_time)

            captura_de_video.grab_frame()
            success, frame = captura_de_video.retrieve_frame()

            frame = draw_grid(frame)
            frame_backup = frame.copy()

            cv2.setMouseCallback("Image", mouse_callback, (frame, frame_backup, K_iterator))

        cv2.imshow("Image", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    captura_de_video.stop()



def draw_grid(img):
    
    for i in range(1, M):
        cv2.line(img, (i * piece_width, 0), (i * piece_width, metadata.get('Height')), (0, 0, 0), 4)
    for i in range(1, N):
        cv2.line(img, (0, i * piece_height), (metadata.get('Width'), i * piece_height), (0, 0, 0), 4)
    
    return img



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

def Create_Kvazaar_ROI(M,N,i, Atlas):

    ROI_Text = []

    #A primeira linha contém quantas divisões existirão no eixo X e em seguida no eixo Y

    QP_Offset = 10
    #Lembre-se: o ROI do Kvazaar é em formato raster
    for x in range(M):
        for y in range(N):
            ROI_Text.append(int(QP_Offset * (1 - Atlas[y, x, i])) )

    #Adciona o numero de Colunas e linhas antes do QP OFFset
    Texto = f"{M} {N}\n{' '.join(map(str, ROI_Text))}"

    file_path = os.path.join(ROI_Lookup_Files_Folder, f'Kvazaar_ROI_Lookup_{i}.txt')

    with open(file_path, 'w') as file:
        file.write(Texto)

def set_Video_360_Metada(media):

    subprocess.run(['exiftool', '-XMP-GSpherical:Spherical="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:Stitched="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:StitchingSoftware="Spherical Metadata Tool', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:ProjectionType="equirectangular', media])



#---------------------------------------------------------------------Classes-------------------------------------------------------------------------------------------------
class Video_Capturing():
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)        

    def grab_frame(self):
        #Primeiro tu dá um grab, ai depois tu da um retrieve
        self.video_capture.grab()

    def retrieve_frame(self):
        return self.video_capture.retrieve()
    
    def set_frame(self, frame_number):
        self.video_capture.set(1, frame_number)
    
    def stop(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------
#Raw video does not contain info about the picture size, so you have to manually provide it. Also, you must provide the correct pixel format

input_yuv_file = "C:\TCC\AR and VR\Videos\SouthGate.yuv"
output_h265_file = "SouthGate.mp4"

#In addition, the aspect ratio of 360-degree immersive video is 2:1, rather than 16:9, so the definition of the resolution is different to traditional
#video, which the 4K resolution is defined as 4096x2048
#pixels and 8K as 8192x4096 pixels.
#liu2017.pdf
'''
subprocess.run(['ffmpeg', '-y' ,'-video_size', '8192x4096' , '-pix_fmt',  'yuv420p' ,'-i', input_yuv_file,
                '-vf',  "scale=1920:960:flags=lanczos", '-c:v', 'hevc', "-x265-params", "lossless=1",
                                                    '-r',  '25', output_h265_file])



subprocess.run(['ffmpeg','-video_size', '8192x4096' , '-pix_fmt',  'yuv420p' ,'-i', input_yuv_file,
                '-vf', 'scale=4096x2048:flags=lanczos', '-pix_fmt', 'yuv420p', 'SouthGateYUV420-4096x2048.yuv'])


'''

                                                

input_video = "SouthGate.mp4"
# 1920x1440
#set_Video_360_Metada(input_video)

metadata = get_video_metadata(input_video)


Diretorio_Atual = os.getcwd()
Intermediate_Files_Folder = os.path.join(Diretorio_Atual, 'Intermediate Files')
ROI_Lookup_Files_Folder = os.path.join(Diretorio_Atual, 'ROI Lookup Files')


#Numero_de_Frames = int(metadata.get('duration') * metadata.get('frame_rate'))
Numero_de_Frames = 25*12


#A variável Gaze Map captura a cada frame o centro da tela para onde a pessoa está olhando
gaze_Map_X = [0 for _ in range(Numero_de_Frames)]
gaze_Map_Y = [0 for _ in range(Numero_de_Frames)]

Gaze_Map = [gaze_Map_X, gaze_Map_Y]

# Divide o vídeo em MxN regioes e cria uma matrix MxN
# "K" é o número de "Gaze Atlas" que queremos criar. Podemos ter um Gaze_Atlas para 0 a 30s e outro para 30 a 60s
# "M" é o número de Colunas
# "N" é o número de Linhas
(M, N, K) = 8,8,6
Gaze_Atlas = np.zeros((M, N, K))

#Tamanho do pedaço da Região de Interesse
piece_width = metadata.get('Width') // M
piece_height = metadata.get('Height') // N
piece_frame_time = Numero_de_Frames // K


#Define o campo de visão a ser usado no vídeo
Campo_de_Visao = 120

Select_Manual_ROI(input_video)

#Automatic_Video(input_video, Numero_de_Frames)

#cProfile.run('Gaze_Atlas = Calculate_Heat_Map(input_video, Gaze_Atlas)')

#Gaze_Atlas = Calculate_Heat_Map(input_video, Gaze_Atlas)

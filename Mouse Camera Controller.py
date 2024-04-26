import vlc
import time
import cv2
import numpy as np

import subprocess #Usado para rodar a linha de comando

def Calculate_Atlas_per_Frame(img, M, N, index_Atlas, Atlas, X_Center, Y_Center):
    #A ideia desta função é apenas atualizar o Atlas

    height, width = img.shape[:2]

    piece_width = width // M
    piece_height = height // N

    #Calcula a distância do centro do Gaze_Map para o centro de cada quadrado  
    for j in range(M):
        for k in range(N):
            center_x = j * piece_width + piece_width // 2
            center_y = k * piece_height + piece_height // 2

            dist_x = min(abs(X_Center - center_x), width - abs(X_Center - center_x))
            #Não existe screen border wrapping no eixo Y

            Atlas[j, k, index_Atlas] += np.sqrt(dist_x**2  + (Y_Center - center_y)**2)       

def Calculate_Heat_Map(video_path, Atlas):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    idx_gaze_Map = 0

    #Se um Campo de Visao de 360 graus está para 1920 pixels
    #Entao, um campo de visao de X graus está para Y = (X * 1920)/360 PIXELS
    FOV_pxls_X = int( (Campo_de_Visao * 1920) / 360)

    #FOV_Y = np.radians(2 * np.arctan(1 / Campo_de_Visao)) -> O VLC calcula o FOV vertical em pixels a partir desta formula
        
    FOV_pxls_Y = int(FOV_pxls_X * metadata.get('Height') / metadata.get('Width'))

    idx_Atlas = 0


    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()

        #As projeçoes equiretangulares não possuem a altura 1080, o que significa que o vídeo possui uma
        #resolução diferente da tela. Por isto, devemos inserir a resolução DA TELA QUE ESTÁ SENDO OBSERVADA
        frame = cv2.resize(frame, (1920, 1080))

        #Atualmente o vídeo só roda até certo tempo. No futuro, se o certo tempo for maior do que a duração do vídeo
        #                                                                                           teremos problemas

        '''
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        '''
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
        

        x_0, y_0 = (Gaze_Map[0][idx_gaze_Map] - FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map]  - FOV_pxls_Y // 2)  
        x_1, y_1 = (Gaze_Map[0][idx_gaze_Map] + FOV_pxls_X // 2, Gaze_Map[1][idx_gaze_Map]  + FOV_pxls_Y // 2) 


        Calculate_Atlas_per_Frame(frame, M, N, idx_Atlas, Atlas, (x_0 + x_1)//2, (y_0 + y_1)//2)

        if (idx_gaze_Map + 1 == Numero_de_Frames):
            break

        if (idx_gaze_Map > ( (Numero_de_Frames // K) * (idx_Atlas + 1) )):
            idx_Atlas += 1


        idx_gaze_Map += 1
    
    
    Distancia_Maxima = np.max(Atlas)
    Atlas = Atlas / Distancia_Maxima
    Atlas = np.round(Atlas, decimals=5)

    return Atlas

def Draw_Atlas(M, N, index_Atlas, Atlas, frame_number):
    #Captura um frame do vídeo e desenha sob ele o Atlas

    video_360.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = video_360.read()
    
    height, width = frame.shape[:2]

    piece_width = width // M
    piece_height = height // N

    for i in range(1, M):
        cv2.line(frame, (i * piece_width, 0), (i * piece_width, height), (0, 0, 0), 2)
    for i in range(1, N):
        cv2.line(frame, (0, i * piece_height), (width, i * piece_height), (0, 0, 0), 2)
        

    for j in range(M):
        for k in range(N):
            center_x = j * piece_width + piece_width // 2
            center_y = k * piece_height + piece_height // 2

            #              cv2.getTextSize(text,                                    font, font_scale, thickness)
            text_size, _ = cv2.getTextSize(str(Atlas[j, k, index_Atlas]), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_width, text_height = text_size
            cv2.putText(frame, str(Atlas[j, k, index_Atlas]), (center_x - text_width // 2, center_y - text_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, (1-Atlas[j, k, index_Atlas])*255), 2, cv2.LINE_AA)

    #A imagem ta saindo com a resolução 1920x1440, que é a resolução original. Por isto, damos um resize nela
    cv2.imwrite(f'Atlas {index_Atlas}.jpg', cv2.resize(frame, (1920, 1080)))
    
    return frame

def Draw_all_Atlases():
    for index_do_Atlas in range(K):
        Frame_Number_Drawn = (Numero_de_Frames // (2 * K)) + index_do_Atlas*(Numero_de_Frames // K)
        Draw_Atlas(M,N,index_do_Atlas, Gaze_Atlas, Frame_Number_Drawn)


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
        #x, y = Gaze_Map[0][i], Gaze_Map[1][i] = 960, 540

        VLC_Ponto_de_Vista.contents.yaw = map_value(x, x_min, x_max, yaw_min, yaw_max) # -> Equivale a rodar a cabeça da esquerda para direita
        VLC_Ponto_de_Vista.contents.pitch =  map_value(y, y_min, y_max, pitch_min, pitch_max) # -> Equivale a rodar a cabeça para cima e para baixo

        vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True)
        #vlc.libvlc_video_update_viewpoint(player, VLC_Ponto_de_Vista, True) -> Se o ultimo parametro for true, ele troca o viewpoint. Se for false, ele soma ou
        #                                                                       subtrai o viewpoint

        time.sleep(1/metadata.get('frame_rate'))

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


def Create_ROI_Lookup_file(width, height,M,N,i, Atlas):

    ROI_Text = []

    piece_width = width // M
    piece_height = height // N

    for x in range(M):
        for y in range(N):
            center_x = x * piece_width
            center_y = y * piece_height

            qp_offset = (2* Atlas[x, y, i]) - 1
            qp_offset = np.round(qp_offset, decimals=5)

            ROI_Text += "addroi=" + str(center_x) + ":" + str(center_y) + ":" + str(piece_width) + ":" + str(piece_height) + ":" + str(qp_offset) + ", "

    ROI_Text = ROI_Text[:-2]#Remove os dois ultimos caracteres ", " para evitar erros

    return ROI_Text

def Write_All_ROI_Lookup(index_Atlas):

    for i in range(index_Atlas):

        Intermed_Text = Create_ROI_Lookup_file(metadata.get('Width'),metadata.get('Height'),M,N,i,Gaze_Atlas)

        Text = ''.join(map(str,Intermed_Text))
        
        # Saving the data to a text file
        with open(f'ROI_LOOKUP_TEXT_{i}.txt', 'w') as file:
            file.write(Text)


#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------

input_video = '18_360_Carnival_of_Venice_Italy_4k_video.mp4'
set_Video_360_Metada(input_video)


video_360 = cv2.VideoCapture(input_video)

metadata = get_video_metadata(input_video)

#Numero_de_Frames = Segundos x Frame/Segundo
Numero_de_Frames = int(metadata.get('duration') * metadata.get('frame_rate'))
#Numero_de_Frames = 24*9

#A variável Gaze Map captura a cada frame o centro da tela para onde a pessoa está olhando
gaze_Map_X = [0 for _ in range(Numero_de_Frames)]
gaze_Map_Y = [0 for _ in range(Numero_de_Frames)]

Gaze_Map = [gaze_Map_X, gaze_Map_Y]

# Divide o vídeo em MxN regioes e cria uma matrix MxN
# "K" é o número de "Gaze Atlas" que queremos criar. Podemos ter um Gaze_Atlas para 0 a 30s e outro para 30 a 60s
(M, N, K) = 8,8,6
Gaze_Atlas = np.zeros((M, N, K))


#Define o campo de visão a ser usado no vídeo
Campo_de_Visao = 90


Automatic_Video(input_video, Numero_de_Frames)

Gaze_Atlas = Calculate_Heat_Map(input_video, Gaze_Atlas) 

#Desenha todos os K Atlas
Draw_all_Atlases()

#Libera o vídeo depois de desenhar o Heat Map sobre ele
video_360.release()

Write_All_ROI_Lookup(K)

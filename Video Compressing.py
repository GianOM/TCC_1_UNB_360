import subprocess
import os#Usado para ler e escrever os arquivos nas pastas corretas



def set_Video_360_Metada(media):

    subprocess.run(['exiftool', '-XMP-GSpherical:Spherical="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:Stitched="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:StitchingSoftware="Spherical Metadata Tool', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:ProjectionType="equirectangular', media])


def get_video_metadata(video_path):

    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate, duration', '-of', 'default=nw=1:nk=1', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    
    metadata = {}
    
    output_lines = result.stdout.strip().split('\n')
    
    metadata['frame_rate'] = eval(output_lines[0])
    #metadata['duracao'] = float(output_lines[1])
    metadata['duracao'] = 43.2

    
    return metadata


#--------------------------------------------------------------Código começa aqui --------------------------------------------------------------------------------------------------
'''
#Faz o scaling de um arquivo h265 para 1920x960
input_h265_file = "C:\TCC\AR and VR\Videos\SJTU 8K 360-Degree Video Sequences H265 Lossless\SiyuanGate.h265"
output_mp4_h265_file = "SiyuanGate.mp4"

subprocess.run(['ffmpeg', '-y' ,'-i', input_h265_file,
                '-vf',  "scale=1920:960:flags=lanczos", '-c:v', 'hevc', "-x265-params", "lossless=1",
                                                    '-r',  '25', '-pix_fmt', 'yuv420p', output_mp4_h265_file])
'''


# Define the input and output video file names
input_video = "SiyuanGate.mp4"
output_video_ROI = 'Compressed_Video_ROI.mp4'
output_video_fixed_QP = 'Compressed_Video.mp4'


Regioes_de_Interesse = []
Quantidade_Gaze_Atlas = 6

Diretorio_Atual = os.getcwd()
Intermediate_Files_Folder = os.path.join(Diretorio_Atual, 'Intermediate Files')
ROI_Lookup_Files_Folder = os.path.join(Diretorio_Atual, 'ROI Lookup Files')


# Lê os arquivos textos que mostram as Regiões de Interesse
for i in range(Quantidade_Gaze_Atlas):

    ROI_file_path = os.path.join(ROI_Lookup_Files_Folder, f'ROI_LOOKUP_TEXT_{i}.txt')

    with open(ROI_file_path, 'r') as file:
        Regioes_de_Interesse.append(file.read())

#       "addroi=ORIGIN_X:ORIGIN_Y:WIDHT:HEIGHT:QOFFSET"
# ROI = "addroi=0:ih/4:iw:ih/2:-1/5" -> O centro da tela possui uma qualidade maior

# Multiplos filtros usam ',' para separar
# https://x265.readthedocs.io/en/stable/ -> Link para os parâmetros disponíveis em '-x265-params'
# https://ffmpeg.org/ffmpeg-filters.html#Video-Filters -> Link para os filtros disponíveis em '-vf' (video filters)


#Ao digitar "ffmpeg -filters" no console revela que o filtro "addroi" NÃO SUPORTA TIMELINE EDITING
# ffmpeg -h filter=addroi

input_videos_after_compression = []
Concat_Video = ""

# Choose a preset. The default is medium. Valid presets are ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, and placebo
Video_General_Preset = 'faster'

Numero_de_Frames = 1080

time_window = (43.2//Quantidade_Gaze_Atlas)

#Cria N divisões originais do vídeo
#Substituir o Range por Quantidade_Gaze_Atlas
for i in range(1):
    
    output_file_ROI = os.path.join(Intermediate_Files_Folder, f"ROI_Encoded_{i}.mp4")
    output_file_Default = os.path.join(Intermediate_Files_Folder, f"Fixed_QP_{i}.mp4")

    '''
    #A forma geral do input_videos_after_compression é : ['-i', "Chunk0.mp4", '-i', "Chunk1.mp4", '-i', "Chunk2.mp4",...]
    input_videos_after_compression.append('-i')
    input_videos_after_compression.append(output_file_ROI)
    Concat_Video += f"[{i}:v] [{i}:a] "
    '''
    #"-ss" é a partir de quantos segundos ele começa a ler o video input stream
    #"-t" é por quanto tempo ele permanece lendo este input video stream
    '''                                                                    
    # --keyint, -I <integer> vira a seguinte sintaxe -> keyint=1
    #Força todos os frames a serem I-Slices se for 1
    #Somente o primeiro frame será um I-slice se for -1
    '''
    # bframes=0 -> 0 bframes consecutivos
    # rd = 2 -> Rate distortion optmizitation. Quanto maior este valor, mais tempo demora-se para codificar
    # CTU = 16 -> Maximum CTU size
    # min-cu-size = 16 -> Minimum CTU Size
    # rc-lookahead = 0
    # frame-threads=1
    #Resultado Atual: 20.125714285714285 ms

    command = [
        "ffmpeg",'-y', "-ss", str(time_window * i), "-t", str(time_window), "-i", input_video,
        "-c:v", "libx265",'-x265-params', f'qpmin=28:qpmax=36:deblock=6: min-cu-size=8:ctu=64:rd=6:bframes=0:rc-lookahead=0:frame-threads=1: psnr=1:ssim=1:csv-log-level=2:csv=Encode_ROI_Log_{i}.csv',
        '-vf', Regioes_de_Interesse[i] , output_file_ROI
    ]
    subprocess.run(command)

    command = [
        "ffmpeg",'-y', "-ss", str(time_window * i), "-t", str(time_window), "-i", input_video,
        '-c:v', 'libx265','-x265-params',f'qp=32:csv-log-level=2:psnr=1:ssim=1:csv=Encode_Fixed_QP_Log_{i}.csv', output_file_Default
    ]
    subprocess.run(command)


'''
#A forma geral do Concat_Video é : [0:v] [0:a] [1:v] [1:a] [2:v] [2:a] [3:v] [3:a]...concat=n=6:v=1:a=1 [v] [a]
Concat_Video += f"concat=n={Quantidade_Gaze_Atlas}:v=1:a=1 [v] [a]"

#Junta os K vídeos que foram divididos e comprimidos usando diferentes ATLAS
comando = ["ffmpeg",'-filter_complex', Concat_Video,
           '-map', "[v]" ,'-map', "[a]","-c:v", "libx265","-preset", Video_General_Preset,
           '-x265-params','csv-log-level=2:csv=Encode_Stat_ROI.csv', output_video_ROI
]
#INSERE O VIDEO INPUTS NA SEGUNDA POSIÇÃO DO ARRAY COMANDO
comando[1:1] = input_videos_after_compression


#Toda vez que rodamos o filtro complexo, mesmo tentando mapear o side data, perdemos o metadado que informa que
#o vídeo é 360 para o video player. Por isto, injetamos todas as tags 360 possíveis no vídeo após sua compressão para tornar o video 360

set_Video_360_Metada(output_video_ROI)
set_Video_360_Metada(output_video_default)

subprocess.run(['ffprobe', input_video])
subprocess.run(['ffprobe', output_video_ROI])
subprocess.run(['ffprobe', output_video_default])
'''
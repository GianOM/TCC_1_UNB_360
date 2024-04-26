import subprocess


def set_Video_360_Metada(media):

    subprocess.run(['exiftool', '-XMP-GSpherical:Spherical="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:Stitched="true', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:StitchingSoftware="Spherical Metadata Tool', media])
    subprocess.run(['exiftool', '-XMP-GSpherical:ProjectionType="equirectangular', media])


def get_video_metadata(video_path):

    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,bit_rate', '-of', 'default=nw=1:nk=1', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    
    metadata = {}
    
    output_lines = result.stdout.strip().split('\n')
    
    metadata['Width'] = int(output_lines[0])
    metadata['Height'] = int(output_lines[1])
    
    # Convert framerate to integer
    metadata['kbit_rate'] = (int(output_lines[2])) / 1000
    
    
    return metadata


# Define the input and output video file names
input_video = '18_360_Carnival_of_Venice_Italy_4k_video.mp4'
output_video_ROI = 'Compressed_Video_ROI.mp4'
output_video_default = 'Compressed_Video.mp4'

#Deprecated function used to get widht, height and bitrate of video
#metadata = get_video_metadata(input_video)


Regioes_de_Interesse = []
Quantidade_Gaze_Atlas = 6
for i in range(Quantidade_Gaze_Atlas):
    # Lê os arquivos txts
    with open(f'ROI_LOOKUP_TEXT_{i}.txt', 'r') as file:

        Regioes_de_Interesse.append(file.read())

#       "addroi=ORIGIN_X:ORIGIN_Y:WIDHT:HEIGHT:QOFFSET"
# ROI = "addroi=0:ih/4:iw:ih/2:-1/5" -> O centro da tela possui uma qualidade maior

# Multiplos filtros usam ',' para separar
# https://x265.readthedocs.io/en/stable/ -> Link para os parâmetros disponíveis em '-x265-params'
# https://ffmpeg.org/ffmpeg-filters.html#Video-Filters -> Link para os filtros disponíveis em '-vf' (video filters)
# Choose a preset. The default is medium. Valid presets are ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, and placebo. Use the slowest preset you have patience for
'''
subprocess.run(['ffmpeg', '-y' ,'-i', input_video,
'-c:v', 'libx265', '-preset', 'faster', '-x265-params', 'aq-mode=4:aq-strength=2:qpmin=25:qpmax=50:csv-log-level=2:csv=Encode_Stat.csv', '-vf', Regioes_de_Interesse[0], '-c:a', 'copy', output_video_ROI,
'-c:v', 'libx265', '-preset', 'faster','-c:a', 'copy', output_video_default
])                          

                                                                      
# --keyint, -I <integer> vira a seguinte sintaxe -> keyint=1
#Força todos os frames a serem I-Slices se for 1
#Somente o primeiro frame será um I-slice se for -1

subprocess.run(['ffmpeg', '-y' ,'-i', input_video, '-c:v', 'libx265', '-preset', 'veryfast',
                                                                      '-x265-params', 'keyint=100:csv-log-level=2:csv=Encode_Stat.csv',
                                                                      '-c:a', 'copy', output_video_default])
'''

#Ao digitar "ffmpeg -filters" no console revela que o filtro "addroi" NÃO SUPORTA TIMELINE EDITING
# ffmpeg -h filter=addroi

input_videos_after_compression = []
Concat_Video = ""

for i in range(Quantidade_Gaze_Atlas):
    start_time = (60//Quantidade_Gaze_Atlas) * i
    output_file = f"Chunk{i}.mp4"

    #A forma geral do input_videos_after_compression é : ['-i', "Chunk0.mp4", '-i', "Chunk1.mp4", '-i', "Chunk2.mp4",...]
    input_videos_after_compression.append('-i')
    input_videos_after_compression.append(output_file)

    Concat_Video += f"[{i}:v] [{i}:a] "

    #"-ss" é a partir de quantos segundos ele começa a ler o video input stream
    #"-t" é por quanto tempo ele permanece lendo este input video stream

    command = [
        "ffmpeg",'-y', "-ss", str(start_time), "-t", str((60//Quantidade_Gaze_Atlas)), "-i", input_video,
        "-c:v", "libx265", "-preset", 'ultrafast', '-x265-params', f'aq-mode=4:aq-strength=2:qpmin=25:qpmax=35:csv-log-level=2:csv=Encode_Stat_{i}.csv',
        '-vf', Regioes_de_Interesse[i],
        "-c:a", "copy", output_file
    ]
    subprocess.run(command)

#A forma geral do Concat_Video é : [0:v] [0:a] [1:v] [1:a] [2:v] [2:a] [3:v] [3:a]...concat=n=6:v=1:a=1 [v] [a]
Concat_Video += f"concat=n={Quantidade_Gaze_Atlas}:v=1:a=1 [v] [a]"

#Junta os K vídeos que foram divididos e comprimidos usando diferentes ATLAS
comando = ["ffmpeg",'-filter_complex', Concat_Video,
           '-map', "[v]" ,'-map', "[a]","-c:v", "libx265","-preset", 'ultrafast',
           '-x265-params','csv-log-level=2:csv=Encode_Stat_ROI.csv', output_video_ROI
]
#INSERE O VIDEO INPUTS NA SEGUNDA POSIÇÃO DO ARRAY COMANDO
comando[1:1] = input_videos_after_compression

subprocess.run(comando)

#"-c:v" é o codec de video e "-c:a" é o codec de audio, que devem ser o próximo parâmetro a ser escolhido
#                         Se dermos um "copy", estaremos copiando para o output o codec de video ou audio
subprocess.run(['ffmpeg', '-y' ,'-i', input_video,
'-c:v', 'libx265', '-preset', 'ultrafast','-x265-params','csv-log-level=2:csv=Encode_Stat_Original.csv','-c:a', 'copy', output_video_default
])    

#Toda vez que rodamos o filtro complexo, mesmo tentando mapear o side data, perdemos o metadado que informa que
#o vídeo é 360 para o video player. Por isto, injetamos todas as tags 360 possíveis no vídeo após sua compressão para tornar o video 360

set_Video_360_Metada(output_video_ROI)
set_Video_360_Metada(output_video_default)

subprocess.run(['ffprobe', input_video])
subprocess.run(['ffprobe', output_video_ROI])
subprocess.run(['ffprobe', output_video_default])
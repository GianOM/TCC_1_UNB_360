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
output_video = 'Compressed_Video.mp4'

#Deprecated function used to get widht, height and bitrate of video
#metadata = get_video_metadata(input_video)

# "addroi=ORIGIN_X:ORIGIN_Y:WIDHT:HEIGHT:QOFFSET"
Regiao_de_Interesse_1 = "addroi=0:0:iw:240:+1, addroi=0:1200:iw:240:+1"

Regiao_de_Interesse_2 = "addroi=0:ih/4:iw:ih/2:-1/5"

# o QP no topo(240 pxls) e em baixo(240 pxls)
# Multiplos filtros usam ',' para separar
# https://x265.readthedocs.io/en/stable/ -> Link para os parâmetros disponíveis em '-x265-params'
# https://ffmpeg.org/ffmpeg-filters.html#Video-Filters -> Link para os filtros disponíveis em '-vf' (video filters)
# Choose a preset. The default is medium. Valid presets are ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, and placebo. Use the slowest preset you have patience for

subprocess.run(['ffmpeg', '-y' ,'-i', input_video, '-c:v', 'libx265', '-preset', 'veryfast',
                                                                      '-x265-params', 'qpmin=30:qpmax=40:qp-adaptation-range=6:csv-log-level=2:csv=Encode_Stat.csv',
                                                                      '-vf', Regiao_de_Interesse_2,
                                                                      '-c:a', 'copy', output_video])

#Ao digitar "ffmpeg -filters" no console revela que o filtro "addroi" NÃO SUPORTA TIMELINE EDITING
# ffmpeg -h filter=addroi


#Toda vez que rodamos o filtro complexo acima, mesmo tentando mapear o side data, perdemos o metadado
#que informa que o vídeo é 360. Por isto, injetamos todas as tags 360 no vídeo

#Força o Output a se tornar um video 360

set_Video_360_Metada(output_video)


subprocess.run(['ffprobe', input_video])

subprocess.run(['ffprobe', output_video])

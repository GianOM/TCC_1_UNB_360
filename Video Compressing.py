import subprocess

# Define the input and output video file names
input_video = '18_360_Carnival_of_Venice_Italy_4k_video.mp4'
output_video = 'Compressed_Video'


#ffmpeg -i input.mp4 -vf "crop= output_widht : output_height : Origin_x : Origin_Y" output.mp4

#In recent ffmpeg versions the spherical packet side-data is supported but to write it in MP4 you have to set the standard compliance mode to unofficial or experimental:
#               https://stackoverflow.com/questions/44760588/preserving-side-data-information-for-360-video-transcoding-using-ffmpeg
# Divide o vídeo em 3 regiões
subprocess.run(['ffmpeg', '-i', input_video, '-vf', "crop=1920:120:0:0", 'Top.mp4'])

subprocess.run(['ffmpeg', '-i', input_video, '-vf', "crop=1920:720:0:120", 'Middle.mp4'])

subprocess.run(['ffmpeg', '-i', input_video, '-vf', "crop=1920:120:0:840", 'Bottom.mp4'])



#Diminui o bitrate dos vídeos
subprocess.run(['ffmpeg', '-i', 'Top.mp4', '-b:v', '16k', '-bufsize', '16k', 'Top_modified.mp4'])

subprocess.run(['ffmpeg', '-i', 'Bottom.mp4', '-b:v', '16k', '-bufsize', '16k', 'Bottom_modified.mp4'])


#subprocess.run(['ffmpeg', '-i', Baixo_esquerda, '-i', Baixo_direita,'-strict', 'unofficial', '-filter_complex', "hstack=inputs=2", 'bottom.mp4'])
subprocess.run(['ffmpeg',
                '-i', 'Top_modified.mp4',
                '-i', 'Middle.mp4',
                '-i', 'Bottom_modified.mp4',
                '-filter_complex', "vstack=inputs=3",
                output_video])


#Toda vez que rodamos o filtro complexo acima, mesmo tentando mapear o side data, perdemos o metadado que informa que o vídeo é 360. Por isto, copiamos todas as tags do video original

subprocess.run(['ffprobe', input_video])

subprocess.run(['ffprobe', output_video])

subprocess.run(['exiftool', '-XMP-GSpherical:Spherical="true', output_video])
subprocess.run(['exiftool', '-XMP-GSpherical:Stitched="true', output_video])
subprocess.run(['exiftool', '-XMP-GSpherical:StitchingSoftware="Spherical Metadata Tool', output_video])
subprocess.run(['exiftool', '-XMP-GSpherical:ProjectionType="equirectangular', output_video])

subprocess.run(['ffprobe', output_video])
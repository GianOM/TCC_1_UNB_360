import subprocess
import os#Usado para ler e escrever os arquivos nas pastas corretas

#A função deste script é apenas comprimir provisoriamente o arquivo do container h265 para que o vídeo possa ser assistido a um framerate aceitável

input_h265_file = "C:\TCC\AR and VR\Videos\SJTU 8K 360-Degree Video Sequences H265 Lossless\SiyuanGate.h265"
output_mp4_h265_file = "SiyuanGate.mp4"

subprocess.run(['ffmpeg', '-y' ,'-i', input_h265_file,
                '-vf',  "scale=1920:960:flags=lanczos", '-c:v', 'hevc', "-x265-params", "lossless=1",
                                                    '-r',  '25', '-pix_fmt', 'yuv420p', output_mp4_h265_file])
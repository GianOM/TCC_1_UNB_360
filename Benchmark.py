import subprocess

#"C:\TCC\AR and VR\Videos\SJTU 8K 360-Degree Video Sequences H265 Lossless\Runners.yuv"


#   '--period', '0',    Period of Intra Pictures(0 means only the first picture is intra)


#TO DO: Entender como funcion as Tiles

#Compressing Using my Kvazaar version
subprocess.run([r"D:\kvazaar\out\build\x64-release\kvazaar",
                '-i' , "SouthGateYUV420-4096x2048.yuv", '--input-res', '4096x2048',
                '--tiles', '8x8',
                '--owf', '5', #Frame-level parallelism (process N+1 frames at a time)
                '--gop', '8', #8 é mais rápido que 0, porém introduz B-Slices???
                '--qp', '28','--subme', '0',
                '--ref','1', # 1 Frames de referencia
                '--no-fast-bipred', '--no-bipred', #No bidirection prediction
                '--period', '16', # Period of Intra Pictures(0 means all intra). 15 Evita propagação de erro
                '--sao', 'off', # Sample Adaptive Offset (off, full)
                '--no-rdoq', # Rate-distortion optimized quantization
                '--rd', '0', # Mode search complexity
                '--fast-residual-cost', '0', # Skip CABAC cost for residual coefficients when QP is below the limit.
                '--me-early-termination', 'sensitive', # Motion estimation termination: off, on, sensitive(o mais rápido)
                '--roi', 'Saliency Map.txt', 
                '--output', 'Output_My_Kvazaar.hevc'])


#Compressing Using Default Kvazaar
subprocess.run([".\kvazaar", '-i' ,"SouthGateYUV420-4096x2048.yuv", '--input-res', '4096x2048',
                '--tiles', '8x8',
                '--owf', '5', #Frame-level parallelism (process N+1 frames at a time)
                '--gop', '8', #8 é mais rápido que 0, porém introduz B-Slices???
                '--qp', '28', '--subme', '0', # Fractional pixel motion estimation level (0: Integer motion estimation only)
                '--ref','1', # 1 Frames de referencia
                '--no-fast-bipred', '--no-bipred', #No bidirection prediction
                '--period', '16', # Period of Intra Pictures(0 means all intra). 15 Evita propagação de erro
                '--sao', 'off', # Sample Adaptive Offset (off, full)
                '--no-rdoq', # Rate-distortion optimized quantization
                '--rd', '0', # Mode search complexity
                '--fast-residual-cost', '0', # Skip CABAC cost for residual coefficients when QP is below the limit.
                '--me-early-termination', 'sensitive', #Motion estimation termination
                '--roi','Saliency Map.txt',
                '--output', 'Output_Kvazaar_Default.hevc'])

import subprocess

#"C:\TCC\AR and VR\Videos\SJTU 8K 360-Degree Video Sequences H265 Lossless\Runners.yuv"


subprocess.run([".\kvazaar", '-i' ,'C:\TCC\AR and VR\Videos\SJTU 8K 360-Degree Video Sequences H265 Lossless\Runners.yuv', '--input-res', '3840x2160', '--gop', '0',
                 '--qp', '28', '--subme', '0', # Fractional pixel motion estimation level (0: Integer motion estimation only)
                 '--ref','3', # 3 Frames de referencia
                 '--no-signhide', '--sao', 'off',
                 '--pu-depth-intra', '0-1', '--pu-depth-inter', '0-1', # Limita a Partition Unit ao range de 64x64 - 32x32
                 '--no-rdoq', '--no-bipred',
                 '--period', '0', # Period of Intra Pictures(0 means only the first picture is intra)
                 '--roi','Kvazaar_ROI_Lookup_0.txt',
                 '--output', 'out.hevc'])


#   '--period', '0',    Period of Intra Pictures(0 means only the first picture is intra)
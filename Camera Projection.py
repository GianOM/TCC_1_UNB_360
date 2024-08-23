import numpy as np
import math
from PIL import Image
import cv2



class Vector_3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z



def setProjectionMatrix(angle_of_view, near_clipping_plane, far_clipping_plane, MATRIX):
    scale = 1/np.tan(angle_of_view * 0.5 * math.pi / 180)
    MATRIX[0][0] = scale
    MATRIX[1][1] = scale
    MATRIX[2][2] = -far_clipping_plane/(far_clipping_plane - near_clipping_plane)
    MATRIX[3][2] = (-far_clipping_plane*near_clipping_plane)/(far_clipping_plane - near_clipping_plane)
    MATRIX[2][3] = -1    

def multPointMatrix(Vector_in, Vector_out, MATRIX):
    Vector_out.x = Vector_in.x * MATRIX[0][0] + Vector_in.y * MATRIX[1][0] + Vector_in.z * MATRIX[2][0] + MATRIX[3][0]

    Vector_out.y = Vector_in.x * MATRIX[0][1] + Vector_in.y * MATRIX[1][1] + Vector_in.z * MATRIX[2][1] + MATRIX[3][1]

    Vector_out.z = Vector_in.x * MATRIX[0][2] + Vector_in.y * MATRIX[1][2] + Vector_in.z * MATRIX[2][2] + MATRIX[3][2] 

    w = Vector_in.x * MATRIX[0][3] + Vector_in.y * MATRIX[1][3] + Vector_in.z * MATRIX[2][3] + MATRIX[3][3]

    if w != 1:
        Vector_out.x = Vector_out.x / w
        Vector_out.y = Vector_out.y / w
        Vector_out.z = Vector_out.z / w

path = r"C:\Users\gianv\OneDrive\Desktop\Projeto TCC\Intermediate Files\Atlas 0 to 240 frames.bmp"
img = cv2.imread(path, 0) 

Height_Output = 1080
Width_Output = 1920

# THETA is left/right angle, PHI is up/down angle, both in degree
THETA = 0
PHI = 0


equ_h, equ_w = img.shape[:2]


#Calcula centro da imagem
equ_cx = (equ_w) / 2.0
equ_cy = (equ_h) / 2.0

#Campo de visão Horizontal e Vertical(em graus), sendo o vertical derivado do aspect ratio
wFOV = 150
hFOV = (float(Height_Output) / Width_Output) * wFOV

w_len = np.tan(np.radians(wFOV / 2.0))
h_len = np.tan(np.radians(hFOV / 2.0))


x_map = np.ones([Height_Output, Width_Output], np.float32)
y_map = np.tile(np.linspace(-w_len, w_len,Width_Output), [Height_Output,1])
z_map = -np.tile(np.linspace(-h_len, h_len,Height_Output), [Width_Output,1]).T

D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)

y_axis = np.array([0.0, 1.0, 0.0], np.float32)
z_axis = np.array([0.0, 0.0, 1.0], np.float32)
[R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
[R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

xyz = xyz.reshape([Height_Output * Width_Output, 3]).T
xyz = np.dot(R1, xyz)
xyz = np.dot(R2, xyz).T
lat = np.arcsin(xyz[:, 2])
lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

lon = lon.reshape([Height_Output, Width_Output]) / np.pi * 180
lat = -lat.reshape([Height_Output, Width_Output]) / np.pi * 180

lon = lon / 180 * equ_cx + equ_cx
lat = lat / 90  * equ_cy + equ_cy


persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

cv2.imshow('image',persp)
cv2.waitKey(0)
cv2.destroyAllWindows()
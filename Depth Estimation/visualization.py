import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02 
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02
image_id = '000120'
rgb_image_dir = './nyuv2/train/{}_color.jpg'.format(image_id)
depth_map = './nyuv2/train/{}_depth.png'.format(image_id)

rgb = cv2.imread(rgb_image_dir)

depth = cv2.imread(depth_map, -1) 
depth = depth.astype(np.float)
depth_x = np.zeros((1,depth.shape[0]*depth.shape[1]))
depth_y = np.zeros((1,depth.shape[0]*depth.shape[1]))
depth_z = np.zeros((1,depth.shape[0]*depth.shape[1]))
for i in range(depth.shape[0]):
    for j in range(depth.shape[1]):
        depth_z[0][depth.shape[0]*j+i]=depth[i][j]
        depth_x[0][depth.shape[0]*j+i]=(j-cx)/fx
        depth_y[0][depth.shape[0]*j+i]=(i-cy)/fy
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(depth_x,depth_y,depth_z,c=depth_z)
ax.invert_xaxis()
ax.view_init(elev=90,azim=90)
plt.show()
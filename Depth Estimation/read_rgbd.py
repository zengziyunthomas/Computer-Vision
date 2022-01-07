import cv2
import numpy as np

# uint8 depth in training
image_id = '000000'
rgb_image_dir = './nyuv2/train/{}_color.jpg'.format(image_id)
depth_map = './nyuv2/train/{}_depth.png'.format(image_id)

rgb = cv2.imread(rgb_image_dir)

depth = cv2.imread(depth_map, -1) 
depth = depth.astype(np.float) / 255.*10.
print(depth.shape)
print(depth.min(), depth.max())


# uint16 depth in testing
rgb_image_dir = './nyuv2/test/{}_color.jpg'.format(image_id)
depth_map = './nyuv2/test/{}_depth.png'.format(image_id)

rgb = cv2.imread(rgb_image_dir)
# print(rgb)
depth = cv2.imread(depth_map, -1)
depth = depth.astype(np.float) / 1000.
print(depth.shape)
print(depth.min(), depth.max())
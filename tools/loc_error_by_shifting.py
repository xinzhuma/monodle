import numpy as np
from PIL import Image
from lib.datasets.kitti.kitti_utils import Calibration


image_file = '../data/KITTI/object/training/image_2/000000.png'
image = Image.open(image_file)
calib_file = '../data/KITTI/object/training/calib/000000.txt'
calib = Calibration(calib_file)

img_w, img_h = image.size[0], image.size[1]
src_x, src_y = np.array([img_w/2]), np.array([img_h/2])
delta_x, delta_y = np.array([8]), np.array([6])
new_x, new_y = src_x + delta_x, src_y + delta_y
depth = np.array([60])

src_location = calib.img_to_rect(src_x, src_y, depth).reshape(-1)
new_location = calib.img_to_rect(new_x, new_y, depth).reshape(-1)
delta_location = np.abs(src_location - new_location)
loc_error = np.linalg.norm(delta_location)

print(src_location)
print(new_location)
print(delta_location)
print(loc_error)
# image.show()



if __name__ == '__main__':
    pass
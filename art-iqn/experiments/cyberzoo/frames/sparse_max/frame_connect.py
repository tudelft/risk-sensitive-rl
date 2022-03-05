import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

img_names = [file for file in glob.glob("*.jpg")]
img_names = np.sort(img_names)

num_imgs = (len(img_names))
weight = 1.0/num_imgs
print(num_imgs)
final_image = np.zeros_like(cv2.imread(img_names[0])).astype(float)
img_shape = np.shape(final_image)
max_arr = np.zeros((img_shape[0],img_shape[1],num_imgs))

gray = cv2.cvtColor(cv2.imread(img_names[0]), cv2.COLOR_BGR2GRAY)
images = []

for i, name in enumerate(img_names):
    print("loaded", i)
    max_arr[:,:,i] = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    images.append(cv2.imread(name))

print('image load complete')

for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        print('pixel:',i,j)
        max_idx = np.argmax(max_arr[i,j,:])
        final_image[i,j,:] = images[max_idx][i,j]

final_image = final_image.astype(int)
cv2.imwrite('../frames_max.jpg',final_image)
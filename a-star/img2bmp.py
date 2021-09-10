import numpy as np
import cv2

imgname = 'map2.png'
top = 28
bottom = 378
left = 108
right = 473

img = cv2.imread(imgname)

# cv2.imshow('a', img)
# cv2.waitKey(0)

img = img[top:bottom, left:right]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite( imgname.split('.')[0] + '.bmp' , img)


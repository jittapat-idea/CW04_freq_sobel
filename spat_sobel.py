import numpy as np
import cv2

img = cv2.imread("pic/Tamamo.png",cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img,cv2.CV_64F)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

kernel = np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
kernel2 = np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])
output = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)

# print("[input] type",img.dtype)

# print('[Laplacian] type',laplacian.dtype)
# print('[sobelx] type',sobelx.dtype)
# print('[sobely] type',sobely.dtype)

sobelx = cv2.normalize(sobelx,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
sobely = cv2.normalize(sobelx,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

height, width = sobelx .shape[:2]
print(height)
print(width)
cv2.imshow('3x3',sobelx)
# cv2.imshow('sobely',sobely)
# cv2.imshow('sobelx',sobelx)
# combined_image = cv2.hconcat([sobelx, sobely ])
# cv2.imshow("sobelx vs sobely", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
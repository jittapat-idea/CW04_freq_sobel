import cv2
import numpy as np

img = cv2.imread('pic/Tamamo.png', cv2.IMREAD_GRAYSCALE)

# แปลงภาพเข้าสู่โดเมนความถี่
imgF = np.fft.fft2(img)
imgF_shifted = np.fft.fftshift(imgF)

kernel = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
# kernel = np.array([[-1,-2,-1],
#                   [0,0,0],
#                   [1,2,1]])
# kernal = np.zeros([3,3],dtype=np.float32)
# kernal[0,0] = 1; kernal[0,1] = 0; kernal[0,2] = -1
# kernal[1,0] = 1; kernal[1,1] = 0; kernal[1,2] = -2
# kernal[2,0] = 1; kernal[2,1] = 0; kernal[2,2] = -1


#แนบเติมฟิลเตอร์ Sobel ให้มีขนาดเท่ากับภาพ
padded = np.pad(kernel, [(0, img.shape[0] - 3), (0, img.shape[1] - 3)], mode='constant')
# kernel_resized = cv2.resize(kernel, (imgF.shape[1], imgF.shape[0]))


kernelF = np.fft.fft2(padded)
kernelF_shifted = np.fft.fftshift(kernelF)

#คูณแบบจุดต่อจุดในโดเมนความถี่
freq = kernelF * imgF_shifted

# # แปลงสเกลค่าความเข้มสีเพื่อปรับปรุงการแสดงผล
# magnitude_spectrum = np.log(1 + np.abs(freq))
# min_val = np.min(magnitude_spectrum)
# max_val = np.max(magnitude_spectrum)
# normalized_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)

# height, width = normalized_spectrum.shape[:2]
# print("freq_height = ",height)
# print("freq_width = ",width)

# height2, width2 = img.shape[:2]
# print("spat_height = ",height2)
# print("spat_width = ",width2)

# cv2.imshow("Frequency Domain", normalized_spectrum)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# spat = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)
spat = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

ans = np.abs(np.fft.ifftn(freq)).astype(np.uint8)


spatz = cv2.normalize(spat,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
freq_umat = cv2.normalize(ans,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

combined_image = cv2.hconcat([spatz,  freq_umat])
cv2.imshow('spat vs freq',combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
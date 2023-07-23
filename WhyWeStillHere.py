import cv2
import numpy as np

img = cv2.imread('pic/Tamamo.png', cv2.IMREAD_GRAYSCALE)

# แปลงภาพเข้าสู่โดเมนความถี่
imgF = np.fft.fft2(img)


kernel = np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])


#แนบเติมฟิลเตอร์ Sobel ให้มีขนาดเท่ากับภาพ
padded = np.pad(kernel, [(0, img.shape[0] - 3), (0, img.shape[1] - 3)], mode='constant')

# kernel_resized = cv2.resize(kernel, (imgF.shape[1], imgF.shape[0]))

kernelF_shifted = np.fft.ifftshift(padded)
kernelF = np.fft.fft2(kernelF_shifted)
#kernelF_shifted = np.fft.fftshift(kernelF)

#คูณแบบจุดต่อจุดในโดเมนความถี่
freq =  imgF *  kernelF

# spat = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)
spat = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)

#ans = np.abs(np.fft.ifftn(freq)).astype(np.uint8)
ans = np.fft.ifft2(freq)
output_shifted = np.fft.fftshift(ans)
output = np.real(output_shifted)

spatz = cv2.normalize(spat,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
freq_umat = cv2.normalize(output,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

combined_image = cv2.hconcat([spatz,  freq_umat])
cv2.imshow('spat vs freq',combined_image)
cv2.imwrite('classwork4/spat vs freq.png',combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
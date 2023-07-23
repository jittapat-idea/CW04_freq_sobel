import cv2
import numpy as np

img = cv2.imread('pic/Tamamo.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])

# แนบเติมฟิลเตอร์ Sobel ให้มีขนาดเท่ากับภาพ
padded = np.pad(kernel, [(0, img.shape[0]-3), (0, img.shape[1]-3)], mode='constant')


kernelF_shifted = np.fft.ifftshift(padded)
kernelF = np.fft.fft2(kernelF_shifted)

# kernelF_shifted = np.fft.fftshift(kernelF)

# แปลงสเกลค่าความเข้มสีเพื่อปรับปรุงการแสดงผล
magnitude_spectrum = np.log(1 + np.abs(kernelF ))
min_val = np.min(magnitude_spectrum)
max_val = np.max(magnitude_spectrum)
normalized_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)

height, width = normalized_spectrum.shape[:2]
print(height)
print(width)

# ans = np.abs(np.fft.ifftn(kernelF)).astype(np.uint8)
# แสดงภาพที่แปลงเข้าสู่โดเมนความถี่ที่ปรับปรุงแล้ว
cv2.imshow("Frequency Domain", normalized_spectrum)
cv2.waitKey(0)
cv2.destroyAllWindows()
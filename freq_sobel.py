import cv2
import numpy as np

img = cv2.imread('pic/Tamamo.png', cv2.IMREAD_GRAYSCALE)

#cast data type to float 32 bit
# img = img.astype(np.float32)

# แปลงภาพเข้าสู่โดเมนความถี่
imgF = np.fft.fft2(img)

# เคลื่อนย้ายตำแหน่งค่าความถี่ตามการเรียงลำดับเฟรียร์
imgF_shifted = np.fft.fftshift(imgF)

#find magnitude & phase
# imgReal = np.real(imgF)
# imgIma = np.imag(imgF)
# imgMag = np.sqrt(imgReal**2 + imgIma**2)
# imgPhs = np.arctan2(imgIma,imgReal)

# imgMag = np.log(1 + imgMag)
# imgMag = cv2.normalize(imgMag,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

# แปลงสเกลค่าความเข้มสีเพื่อปรับปรุงการแสดงผล
magnitude_spectrum = np.log(1 + np.abs(imgF_shifted))
min_val = np.min(magnitude_spectrum)
max_val = np.max(magnitude_spectrum)
normalized_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)

height, width = normalized_spectrum.shape[:2]
print(height)
print(width)
# ans = np.abs(np.fft.ifftn(imgF_shifted)).astype(np.uint8)

# แสดงภาพที่แปลงเข้าสู่โดเมนความถี่ที่ปรับปรุงแล้ว
cv2.imshow("Frequency Domain", normalized_spectrum )
cv2.waitKey(0)
cv2.destroyAllWindows()







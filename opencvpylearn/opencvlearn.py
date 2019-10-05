import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def magnitude_spectrum_picture():
    img = cv.imread('../Project1/lena_top.jpg',0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()






img = cv.imread('../Project1/lena_top.jpg',0)

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))




rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-50:crow+50, ccol-50:ccol+50] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

mask5 = np.zeros((rows,cols,2),np.uint8)
mask5[crow-15:crow+15, ccol-15:ccol+15] = 1

# apply mask and inverse DFT
fshift5 = dft_shift*mask5
f_ishift5 = np.fft.ifftshift(fshift5)
img_back5 = cv.idft(f_ishift5)
img_back5 = cv.magnitude(img_back5[:,:,0],img_back5[:,:,1])


mask150 = np.zeros((rows,cols,2),np.uint8)
mask150[crow-100:crow+100, ccol-100:ccol+100] = 1

# apply mask and inverse DFT
fshift150 = dft_shift*mask150
f_ishift150 = np.fft.ifftshift(fshift150)
img_back150 = cv.idft(f_ishift150)
img_back150 = cv.magnitude(img_back150[:,:,0],img_back150[:,:,1])



masksharp = np.ones((rows,cols,2),np.uint8)
masksharp[crow-100:crow+100, ccol-100:ccol+100] = 0

# apply mask and inverse DFT
fshiftsharp = dft_shift*masksharp
f_ishiftsharp = np.fft.ifftshift(fshiftsharp)
img_backsharp = cv.idft(f_ishiftsharp)
img_backsharp = cv.magnitude(img_backsharp[:,:,0],img_backsharp[:,:,1])



plt.figure(figsize=(20,8))

plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(img_back5, cmap = 'gray')
plt.title('Magnitude Spectrum5 '), plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum50 '), plt.xticks([]), plt.yticks([])

plt.subplot(235),plt.imshow(img_back150, cmap = 'gray')
plt.title('Magnitude Spectrum 150 '), plt.xticks([]), plt.yticks([])

plt.subplot(236),plt.imshow(img_backsharp, cmap = 'gray')
plt.title('Magnitude sharp 150 '), plt.xticks([]), plt.yticks([])
plt.show()
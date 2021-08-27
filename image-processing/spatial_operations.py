import numpy as np 
import cv2
import matplotlib.pyplot as plt 

# source image 
lena = cv2.imread('..\sources\lena.png')
r, c, i = lena.shape

# make a noise on the picture 
noise_lena = lena + np.random.normal(0, 20, (r, c, i)).astype(np.uint8)

# helper method for plotting source image and the ouput image  
def plotImg(source, output, title_output, title_original="Source"):
    plt.figure(figsize = (10,10))
    plt.subplot(121)
    plt.title(title_original)
    plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title(title_output)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.show()

# mean filtering 
mean_kernel = np.ones((5,5))/25
filtered_lena = cv2.filter2D(noise_lena, ddepth = -1, kernel = mean_kernel)# filter2D does convolution between source and kernel 
plotImg(noise_lena, filtered_lena, "Mean Filter")

# gaussian blur 
gaussian_lena = cv2.GaussianBlur(noise_lena, (5, 5), sigmaX=4, sigmaY=4)
plotImg(noise_lena, gaussian_lena, "Gaussian Filter")

# sharpening image
sharpening_kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

sharpened_lena = cv2.filter2D(lena, ddepth = -1, kernel = sharpening_kernel)
plotImg(lena, sharpened_lena, "Sharpening Filter")

# sobel filter for edge detection 
gray_lena = cv2.imread('..\sources\lena.png', cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(gray_lena, -1, dx = 1, dy = 0, ksize = 3)
sobel_y = cv2.Sobel(gray_lena, -1, dx = 0, dy = 1, ksize = 3)
plt.figure(figsize = (10,10))
plt.subplot(121)
plt.title("Sobel Filter X")
plt.imshow(sobel_x)
plt.subplot(122)
plt.title("Sobel Filter Y")
plt.imshow(sobel_y)
plt.show() 

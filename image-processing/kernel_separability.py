import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def genGaussianKernel2D(width, sigma):
    kernel_2d = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            kernel_2d[i][j] = (1/(2*np.pi*sigma**2)) * np.exp(((-1*((i-(width-1)/2)**2+(j-(width-1)/2)**2))/(2*sigma**2)))
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d 

def genGaussianKernel1D(length, sigma):
    kernel_1d = np.zeros(length)
    for i in range(length):
        kernel_1d[i] = (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(((-1*(i-(length-1)/2.)**2) / (2*sigma**2.)))
    kernel_1d = kernel_1d/np.sum(kernel_1d)
    kernel_1d = np.array([kernel_1d])
    return kernel_1d

def addGaussianNoise(img, mean, std, seed=0):
    np.random.seed(seed)
    img_noise = img.copy() 
    img_noise = img.astype(np.float32)/255.
    noise = np.random.normal(mean, std, img.shape)
    img_noise = img_noise + noise
    img_noise = np.clip(img_noise, 0., 1.)
    img_noise = (img_noise*255).astype(np.uint8)
    return img_noise

img = cv2.imread('../sources/lena512.bmp')

width, sigma = 11, 3

img_gnoise = addGaussianNoise(img, 0, 0.1)
kernel_2d = genGaussianKernel2D(width, sigma)
kernel_x = genGaussianKernel1D(width, sigma)
kernel_y = np.transpose(genGaussianKernel1D(width, sigma))


img_kernel1d_x = cv2.filter2D(img_gnoise, -1, kernel_x)
img_kernel1d_xy = cv2.filter2D(img_kernel1d_x, -1, kernel_y)
img_kernel2d = cv2.filter2D(img_gnoise, -1, kernel_2d)

plt.figure(figsize = (5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gnoise, 'gray')
plt.title("original noised image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_kernel1d_x, 'gray')
plt.title("1d filtered in x direction")
plt.axis("off")

plt.figure(figsize = (5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_kernel1d_xy, 'gray')
plt.title('1d filtered (x->y)')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_kernel2d, 'gray')
plt.title('2d filtered')
plt.axis("off")

# Compute the difference array here
img_diff = np.abs((img_kernel1d_xy.astype(np.float32)
       -img_kernel2d.astype(np.float32))).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_diff, 'gray', vmin=0, vmax=255)
plt.title('Difference image\n Max abs difference={0:d}'
                        .format(np.max(img_diff)))
plt.axis("off")
plt.show()

def cal_1d_FLOPs(img, kernel):
  FLOPs = 0
  r, c = img.shape[:2]
  r_kernel, c_kernel = kernel.shape
  padding = max(r_kernel, c_kernel)//2
  num_of_multiplication = r_kernel * c_kernel
  num_of_addition = r_kernel * c_kernel - 1
  if max(r_kernel, c_kernel) == r_kernel:
    FLOPs = (r+2*padding-r_kernel+1)*(c)*(num_of_multiplication + num_of_addition)
  else:
    FLOPs = (c+2*padding-c_kernel+1)*(r)*(num_of_multiplication + num_of_addition)

  return FLOPs

def cal_2d_FLOPs(img, kernel):
  FLOPs = 0
  r, c = img.shape[:2]
  r_kernel, c_kernel = kernel.shape
  padding = r_kernel//2
  num_of_multiplication = r_kernel * c_kernel
  num_of_addition = r_kernel * c_kernel - 1
  FLOPs= (r+2*padding-r_kernel+1)*(c+2*padding-c_kernel+1)*(num_of_multiplication + num_of_addition)

  return FLOPs

FLOPs_x = cal_1d_FLOPs(img_gnoise, kernel_x)
FLOPs_y = cal_1d_FLOPs(img_kernel1d_x, kernel_y)
FLOPs_1D = FLOPs_x + FLOPs_y
print('1D Filter FLOPs:', FLOPs_1D)
FLOPs_2D = cal_2d_FLOPs(img_gnoise, kernel_2d)
print('2D Filter FLOPs:', FLOPs_2D)
import numpy as np 
import cv2
import matplotlib.pyplot as plt 

# source image 
lena = cv2.imread('..\sources\lena.png')

# helper method for plotting source image and the ouput image  
def plotImg(source, output, title_output, title_original="Source"):
    plt.figure(figsize = (5,5))
    plt.subplot(121)
    plt.title(title_original)
    plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title(title_output)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.show()

# scaling by x
x_scaled_lena = cv2.resize(lena, None, fx = 3, fy = 1, interpolation = cv2.INTER_CUBIC)
plotImg(lena, x_scaled_lena, "Scaled by X")

# scaling by y 
y_scaled_lena = cv2.resize(lena, None, fx = 1, fy = 3, interpolation = cv2.INTER_CUBIC)
plotImg(lena, y_scaled_lena, "Scaled by Y")

# translation 
x, y = 50, 50
transformation_matrix = np.float32([[1, 0, x], [0, 1, y]])
r, c, _ = lena.shape
translated_lena = cv2.warpAffine(lena, transformation_matrix, (c + x, r + y))
plotImg(lena, translated_lena, "Translation")

# rotation 
angle = 45
rotation_matrix = cv2.getRotationMatrix2D(center = (c // 2 - 1, r // 2 - 1), angle = angle, scale = 1)
rotated_lena = cv2.warpAffine(lena, rotation_matrix, (c, r))
plotImg(lena, rotated_lena, "Rotation by 45 degrees")


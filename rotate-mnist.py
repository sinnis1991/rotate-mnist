from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

X, y = loadlocal_mnist(
        images_path='./train-images-idx3-ubyte', 
        labels_path='./train-labels-idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])


img = np.reshape(X[0],(28,28))
print (len(img.shape))

imgplot = plt.imshow(img,cmap = 'gray', interpolation = 'bicubic')
plt.show()

def rotate_image(X,theta):
	r_image = np.zeros(X.shape)
	return r_image

img2 = rotate_image(img,0)
imgplot2 = plt.imshow(img2,cmap = 'gray', interpolation = 'bicubic')
plt.show()
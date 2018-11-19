from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import scipy.misc
import os
import make_gif

class rotate_mnist(object):

	#build model
    def __init__(self, x_path, y_path):
    	self.x_path = x_path
    	self.y_path = y_path
    	self.X, self.y = loadlocal_mnist(
                     images_path = x_path, 
                     labels_path = y_path)

    def show_example(self):
    	img = np.reshape(self.X[0],(28,28))
    	print (len(img.shape))
    	imgplot = plt.imshow(img,cmap = 'gray', interpolation = 'bicubic')
    	plt.show()
    
    def show_rotate_example(self, theta):
    	example = np.reshape(self.X[0],(28,28))
    	im = np.zeros(example.shape)
    	for i in range(28):
    		for j in range(28):
    			x_ = i-14
    			y_ = j-14
    			# x_ = x*cos - y*sin
    			# y_ = x*sin + y*cos
    			# ==> x = x_*cos + y_*sin
    			#     y = y_*cos - x_*sin
    			x = round(x_*math.cos(theta) + y_*math.sin(theta)+14)
    			y = round(y_*math.cos(theta) - x_*math.sin(theta)+14)
    			if x>=0 and x<28 and y>=0 and y<28:
    				im[i,j] = example[x,y]
    	imgplot = plt.imshow(im,cmap = 'gray', interpolation = 'bicubic')
    	plt.show()	

    def rotate(self,X,theta):
    	im = np.zeros(X.shape)
    	for i in range(28):
    		for j in range(28):
    			x_ = i-14
    			y_ = j-14
    			x = round(x_*math.cos(theta) + y_*math.sin(theta)+14)
    			y = round(y_*math.cos(theta) - x_*math.sin(theta)+14)
    			if x>=0 and x<28 and y>=0 and y<28:
    				im[i,j] = X[x,y]
    	return im

    def gif_example(self):
    	row = 8
    	col = 8
    	path = './gif_images'
    	if not os.path.exists(path):
    	    os.mkdir(path)
    	for i in range(12):
    		im = np.zeros((28*row,28*col))
    		for m in range(row):
    			for n in range(col):
    				X = np.reshape(self.X[row*m+n],(28,28))
    				X = self.rotate(X,math.pi/12*i)
    				im[28*m:28*(m+1),28*n:28*(n+1)] = X
    		# imgplot = plt.imshow(im,cmap = 'gray', interpolation = 'bicubic')
    		# plt.show()	
    		scipy.misc.imsave(os.path.join(path,'%s.jpg' % i), im)



    			


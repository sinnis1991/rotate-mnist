from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import scipy.misc
import os
# import make_gif

class rotate_mnist(object):

	#build model
    def __init__(self, x_path, y_path):
    	self.x_path = x_path
    	self.y_path = y_path
    	self.X, self.y = loadlocal_mnist(
                     images_path = x_path, 
                     labels_path = y_path)

    	self.X = np.reshape(self.X,(-1,28,28,1)).astype(np.float)

    	angle = np.tile(np.arange(12),len(self.X))*math.pi/12.
    	index = np.arange(len(self.X)).repeat(12)
    	seed = 547
    	np.random.seed(seed)
    	np.random.shuffle(angle)
    	np.random.seed(seed)
    	np.random.shuffle(index)
    	self.angle = angle
    	self.index = index



    def show_example(self):
    	img = np.reshape(self.X[0],(28,28))
    	print (img.shape)
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

    def test(self):
    	print(self.angle[:100])
    	print(self.index[:100])

    def getNext_batch(self,iter_num=0, batch_size=64):

    	ro_num = len(self.index) / batch_size - 1
    	
    	if iter_num % ro_num == 0:
        
            length = len(self.index)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.angle = np.array(self.angle)
            self.angle = self.angle[perm]
            self.index = np.array(self.index)
            self.index = self.index[perm]

    	angle = self.angle[int(iter_num % ro_num) * batch_size: int(iter_num%ro_num + 1) * batch_size]
    	index = self.index[int(iter_num % ro_num) * batch_size: int(iter_num%ro_num + 1) * batch_size]
    	X = np.zeros((batch_size,28,28,1)).astype(np.float)
    	y = np.zeros(batch_size)
    	for i in range(batch_size):
        	X[i] = self.rotate(self.X[index[i]],angle[i])
        	y[i] = self.y[index[i]]
        
    	return X,angle,y

    def batch_test(self,iter):
    	X,a,y = self.getNext_batch(iter_num=iter)

    	for i in range(len(X)):

    		im = np.reshape(X[i],(28,28))	
    		print("angle: %s  lable: %s " % (np.round(a[i]/math.pi*12.),y[i]))
    		imgplot = plt.imshow(im,cmap = 'gray', interpolation = 'bicubic')
    		plt.show()



    			


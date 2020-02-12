import numpy as np
import cv2
import skimage.transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gaussian2D(nsig=9, kernel_length=25):
	"""Generates the 2D gaussian kernel"""
	# We take kernel_length because we lose one value when we use np.diff()
	x = np.linspace(-nsig, nsig, kernel_length)
	# Cumulative normal distribution function
	kernel1 = np.sqrt(1/(2*np.pi*nsig))*np.exp(-0.5*(x**2)/nsig) #st.norm.cdf(x)
	fig = plt.figure()
	plt.plot(kernel1)
	plt.savefig('./DoG_filters/kernel.png')
	# Calculate outer product
	kernel2 = np.outer(kernel1, kernel1)
	# Dividing by the sum to normalize the kernel
	return kernel2/kernel2.sum()

def gaussian_first_derivative(scale, rotated_points, kernel_length):

	var_x = scale**2
	var_y = (3*scale)**2
	mean = 0
	x = rotated_points[0,:]
	y = rotated_points[1,:]
	gaussian_x = (1/np.sqrt(2*np.pi*var_x))*(np.exp((-1*(x-mean)**2)/(2*var_x)))
	gaussian_y = (1/np.sqrt(2*np.pi*var_y))*(np.exp((-1*(y-mean)**2)/(2*var_y)))
	first_derivative_x = -gaussian_x*((x-mean)/(var_x))
	# Gaussian can be denoted using the convolution of gaussian in x with 
	# gaussian in y. Thus, using theory of convolution here d(x*y) = 
	# (dx)*y where * is the convolution and d(x) is the derivative of x
	image = first_derivative_x*gaussian_y
	image = np.reshape(image, [kernel_length, kernel_length])
	return image

def gaussian_second_derivative(scale, rotated_points, kernel_length):

	var_x = scale**2
	var_y = (3*scale)**2
	mean = 0
	x = rotated_points[0,:]
	y = rotated_points[1,:]
	gaussian_x = (1/np.sqrt(2*np.pi*var_x))*(np.exp((-1*(x-mean)**2)/(2*var_x)))
	gaussian_y = (1/np.sqrt(2*np.pi*var_y))*(np.exp((-1*(y-mean)**2)/(2*var_y)))
	second_derivative_x = gaussian_x*(((x-mean)**2 - var_x)/(var_x**2))
	# Gaussian can be denoted using the convolution of gaussian in x with 
	# gaussian in y. Thus, using theory of convolution here d(d(x*y)) = 
	# (d2x)*y where * is the convolution and d(x) is the derivative of x
	image = second_derivative_x*gaussian_y
	image = np.reshape(image, [kernel_length, kernel_length])
	return image

def gaussian2D_lm(x, y, scale):
	var = scale**2
	gaussian = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var))
	return gaussian

def laplace_gaussian2D(x, y, scale):
	var = scale**2
	gaussian = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var))
	double_derivative_gaussian = gaussian*((x*x + y*y) - var)/(var**2)
	return double_derivative_gaussian


def gabor_filters_fn(sigma, theta, Lambda, psi, gamma):

    sigma_x = sigma
    sigma_y = float(sigma) / gamma   
    x, y = np.meshgrid(np.linspace(-7,7,49), np.linspace(-7,7,49))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi * x_theta / sigma + psi)

    return gb

def texton_dog(image_, filter_bank):
	texton_ = np.zeros_like(image_)
	number_filters, height, width = filter_bank.shape
	for i in range(number_filters):
		filtered = cv2.filter2D(image_, -1, filter_bank[i])
		texton_ = np.dstack((texton_, filtered))

	return texton_[:,:,3:]

def texton_lms(image_, filter_bank):
	texton_ = np.zeros_like(image_)
	height, width, number_filters = filter_bank.shape
	for i in range(number_filters):
		filtered = cv2.filter2D(image_, -1, filter_bank[:,:,i])
		texton_ = np.dstack((texton_, filtered))

	return texton_[:,:,3:]

def texton_gabor(image_, filter_bank):
	texton_ = np.zeros_like(image_)
	number_filters = len(filter_bank)
	for i in range(number_filters):
		filtered = cv2.filter2D(image_, -1, filter_bank[i])
		texton_ = np.dstack((texton_, filtered))

	return texton_[:,:,3:]

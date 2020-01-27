import numpy as np
import cv2
import skimage.transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gaussian2D(nsig=9, kernel_length=25):
	"""Generates the 2D gaussian kernel"""
	# We take kernel_length+1 because we lose one value when we use np.diff()
	x = np.linspace(-nsig, nsig, kernel_length+1)
	# Cumulative normal distribution function
	kernel1 = np.sqrt(1/(2*np.pi*nsig))*np.exp(-0.5*(x**2)/nsig) #st.norm.cdf(x)
	fig = plt.figure()
	plt.plot(kernel1)
	plt.savefig('./DoG_filters/kernel.png')
	# out[i] = a[i+1] - a[i]
	# kernel1 = np.diff(kernel1)
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
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))

    # Rotation
    # points = np.array([x.flatten(), y.flatten()], dtype=float)
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], \
				# 						[np.sin(theta), np.cos(theta)]])    
    # rotated_points = np.dot(rotation_matrix, points)
    # x_ = np.reshape(rotated_points[0,:], [-1,-1])
    # y_ = np.reshape(rotated_points[1,:], [-1,-1])
    x_ = x*np.cos(theta) + y*np.sin(theta)
    y_ = -x*np.sin(theta) + y*np.cos(theta)

    gabor = np.exp(-.5 * (x_ ** 2 / sigma_x ** 2 + y_ ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_ + psi)
    return gabor
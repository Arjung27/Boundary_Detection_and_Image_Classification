#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import skimage.transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st

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

def generate_dog_filters(number_orientations, scales, kernel_length=25):
	""" Return an array of DoG filters"""
	orientations = np.linspace(0,360,number_orientations+1)
	dog_kernels = []
	for vals in scales:
		gaussian_kernel = gaussian2D(vals, kernel_length)
		border_type = cv2.borderInterpolate(0, 1, cv2.BORDER_REFLECT)
		sobel_on_gaussian = cv2.Sobel(gaussian_kernel, cv2.CV_64F, 1, 0, \
													borderType=border_type)
		for i, angle in enumerate(orientations[:-1]):
			image_center = tuple(np.array(sobel_on_gaussian.shape[1::-1])/2)
			# print(sobel_on_gaussian[20,20])
			rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
			# dog_kernels.append(skimage.transform.rotate(sobel_on_gaussian, \
			# 														angle))
			dog_kernels.append(cv2.warpAffine(sobel_on_gaussian, rot_mat, \
				sobel_on_gaussian.shape[1::-1], flags=cv2.INTER_CUBIC))

	return dog_kernels

def gaussian_first_derivative(scale, rotated_points, kernel_length):

	var_x = scale**2
	var_y = (3*scale)**2
	mean = 0
	x = rotated_points[0,:]
	y = rotated_points[1,:]
	gaussian_x = (1/np.sqrt(2*np.pi*var_x))*(np.exp((-1*(x-mean)**2)/(2*var_x)))
	gaussian_y = (1/np.sqrt(2*np.pi*var_y))*(np.exp((-1*(y-mean)**2)/(2*var_y)))
	first_derivative_x = -gaussian_x*((x-mean)/(var_x))
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
	image = second_derivative_x*gaussian_y
	image = np.reshape(image, [kernel_length, kernel_length])
	return image

def generate_lm_filter(number_orientations, scales, kernel_length=49):
	
	orientations = np.linspace(0,2*np.pi,number_orientations+1)[0:-1]
	total_filters = 48
	lm_filter_bank = np.zeros([kernel_length, kernel_length, total_filters])
	gaussian_index = (kernel_length - 1) /2
	x = [np.arange(-gaussian_index, gaussian_index + 1)]
	y = [np.arange(-gaussian_index, gaussian_index + 1)]
	[x,y] = np.meshgrid(x,y)
	points = np.array([x.flatten(), y.flatten()], dtype=float)
	count = 0

	for scale in scales[0:-1]:
		for angle in orientations:
			rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], \
										[np.sin(angle), np.cos(angle)]])
			rotated_points = np.dot(rotation_matrix, points)
			lm_filter_bank[:,:,count] = gaussian_first_derivative(scale, \
											rotated_points, kernel_length)
			lm_filter_bank[:,:,count + 18] = gaussian_second_derivative(scale, \
											rotated_points, kernel_length)
			count += 1

	return lm_filter_bank
	

def draw_dog_filters(dog_filter_bank, number_orientations, number_scales):
	alpha = 2 # Should be a perfect divisor of number_orietations
	fig, axs = plt.subplots(int(alpha*number_scales), int(number_orientations/alpha), figsize=(20,20))
	for i in range(len(dog_filter_bank)):
		# axs[int(i/number_orientations), i%number_orientations].plot(dog_filter_bank[i])
		plt.subplot(int(alpha*number_scales), int(number_orientations/alpha), i+1)
		plt.axis('off')
		plt.imshow(dog_filter_bank[i], cmap='gray')
	plt.savefig('./DoG_filters/image.png')
	plt.close()

def draw_lm_filter(lm_filter_bank, number_orientations, number_scales):
	for i in range(0,36):
		plt.subplot(6,6,i+1)
		plt.axis('off')
		plt.imshow(lm_filter_bank[:,:,i], cmap='gray')
	plt.savefig('./lm_filter.png')

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [9, 16]
	number_orientations = 16
	dog_filter_bank = generate_dog_filters(number_orientations, scales, 49)
	draw_dog_filters(dog_filter_bank, number_orientations, len(scales))

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
	number_orientations = 6
	lm_filter_bank = generate_lm_filter(number_orientations, scales, 49)
	draw_lm_filter(lm_filter_bank, number_orientations, len(scales))

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 



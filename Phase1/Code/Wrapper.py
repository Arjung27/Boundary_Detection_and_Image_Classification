#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Arjun Gupta
M.Eng. in Robotics,
University of Maryland, College Park

"""

# Code starts here:

import numpy as np
import cv2
import skimage.transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import filter_util


def generate_dog_filters(number_orientations, scales, kernel_length=25):
	""" Return an array of DoG filters"""
	orientations = np.linspace(0,360,number_orientations+1)
	dog_kernels = []
	for vals in scales:
		gaussian_kernel = filter_util.gaussian2D(vals, kernel_length)
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
			lm_filter_bank[:,:,count] = filter_util.gaussian_first_derivative(scale, \
											rotated_points, kernel_length)
			lm_filter_bank[:,:,count + 18] = filter_util.gaussian_second_derivative(scale, \
											rotated_points, kernel_length)
			count += 1

	# Total images are 18 for 1st derivative and 18 for 2nd derivative => 36
	count = 36
	for scale in scales:
		lm_filter_bank[:, :, count] = filter_util.gaussian2D_lm(np.array(x), np.array(y), scale)
		lm_filter_bank[:, :, 4 + count] = filter_util.laplace_gaussian2D(np.array(x), np.array(y), scale)
		lm_filter_bank[:, :, 8 + count] = filter_util.laplace_gaussian2D(np.array(x), np.array(y), 3*scale)
		count += 1

	return lm_filter_bank


def generate_gabor_filters(scales, theta, Lambda, psi, gamma, number_filters):

	gabor_filter_bank = []
	for scale in scales:
		gabor = filter_util.gabor_filters_fn(scale, theta, Lambda, psi, gamma)
		angle = np.linspace(0, 180, number_filters + 1)
		for i, ang in enumerate(angle[:-1]) :
			image_center = tuple(np.array(gabor.shape[1::-1])/2)
			rot_mat = cv2.getRotationMatrix2D(image_center, ang, 1.0)
			gabor_image = cv2.warpAffine(gabor, rot_mat, gabor.shape[1::-1])
			# print(gabor_image[12:37, 12:37].shape)
			gabor_filter_bank.append(gabor_image[12:37, 12:37])

	return gabor_filter_bank

def half_disk_masks(scales, number_orientations):
	half_disk_masks = []
	angles = np.linspace(0, 180, number_orientations, endpoint=False)
	for scale in scales:
		base = np.zeros([2*scale+1, 2*scale+1])
		cv2.circle(base, (scale, scale), scale, (255,255,255), -1)
		base_copy = base.copy()
		base[:scale, :] = 0
		base_copy[scale:, :] = 0
		for angle in angles:
			rows, cols = base.shape
			rotation = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
			base_copy = cv2.warpAffine(base_copy, rotation, (cols, rows))
			base = cv2.warpAffine(base, rotation, (cols, rows))
			half_disk_masks.append(base_copy)
			half_disk_masks.append(base)
	return half_disk_masks

def texton(image_, dog_filter_bank, lms_filter_bank, lml_filter_bank, \
								gabor_filter_bank, number_clusters, filename):
	texton_dog = filter_util.texton_dog(image_, \
									np.array(dog_filter_bank))
	texton_lms = filter_util.texton_lms(image_, lms_filter_bank)
	# texton_lml = filter_util.texton_fn(image_, lml_filter_bank)
	texton_gabor = filter_util.texton_gabor(image_, \
												gabor_filter_bank)
	concat_text_map = np.dstack((texton_dog, texton_lms,\
												texton_gabor))
	kmeans_input = np.reshape(concat_text_map, [-1,concat_text_map.shape[2]])
	kmeans = KMeans(n_clusters=number_clusters, random_state=2)
	kmeans.fit(kmeans_input)
	labels = kmeans.predict(kmeans_input)
	print(concat_text_map.shape[0], concat_text_map.shape[1])
	texton_map = np.reshape(labels, \
						(concat_text_map.shape[0], concat_text_map.shape[1]))
	figure = plt.figure()
	plt.imshow(texton_map)
	plt.savefig('./texton/final_'+filename+'.png')
	plt.close()

	return texton_map

def brightness(image_, filename, number_clusters):
	h, w = image_.shape
	kmeans_input = np.reshape(image_, [-1, 1])
	kmeans = KMeans(n_clusters=number_clusters, random_state=2)
	kmeans.fit(kmeans_input)
	labels = kmeans.predict(kmeans_input)
	brightness_ = np.reshape(labels,(h,w))
	figure = plt.figure()
	plt.imshow(brightness_)
	plt.savefig('./brightness/'+filename+'.png')
	plt.close()
	return brightness_

def color(image_, filename, number_clusters):
	h, w, c = image_.shape
	kmeans_input = np.reshape(image_, [-1, c])
	kmeans = KMeans(n_clusters=number_clusters, random_state=2)
	kmeans.fit(kmeans_input)
	labels = kmeans.predict(kmeans_input)
	color_ = np.reshape(labels,(h,w))
	figure = plt.figure()
	plt.imshow(color_)
	plt.savefig('./color/'+filename+'.png')
	plt.close()
	return color_
	
def draw_dog_filters(dog_filter_bank, number_orientations, number_scales, filename="DoG_filters"):

	alpha = 2 # Should be a perfect divisor of number_orietations
	fig, axs = plt.subplots(int(alpha*number_scales), int(number_orientations/alpha), figsize=(20,20))
	for i in range(len(dog_filter_bank)):
		# axs[int(i/number_orientations), i%number_orientations].plot(dog_filter_bank[i])
		plt.subplot(int(alpha*number_scales), int(number_orientations/alpha), i+1)
		plt.axis('off')
		plt.imshow(dog_filter_bank[i], cmap='gray')
	plt.savefig('./DoG_filters/'+filename+'.png')
	plt.close()

def draw_lm_filter(lm_filter_bank, number_orientations, number_scales, filename):

	for i in range(0,48):
		plt.subplot(4,12,i+1)
		plt.axis('off')
		plt.imshow(lm_filter_bank[:,:,i], cmap='gray')
	plt.savefig('./lm_filter/'+filename+'.png')
	plt.close()

def draw_gabor_filter(gabor_filter_bank, rows, cols, filename):

	for i in range(0, len(gabor_filter_bank)):
		plt.subplot(rows, cols, i+1)
		plt.axis('off')
		plt.imshow(gabor_filter_bank[i], cmap='gray')
	plt.savefig('./gabor_filter/'+filename+'.png')
	plt.close()

def plot_half_disk(half_disk_masks, rows, cols):
	for i in range(len(half_disk_masks)):
		plt.subplot(rows, cols, i+1)
		plt.axis('off')
		plt.imshow(half_disk_masks[i], cmap='gray')
	plt.savefig('./half_disk_masks/masks.png')
	plt.close()

def chi_sqr_distance(image, number_bins, left_mask, right_mask):
	chi_sqr_distance = image*0
	epsilon = np.exp(10**-7)
	for i in range(number_bins):
		temp = image.copy()
		temp[image == i] = 1
		temp[image != i] = 0
		g_i = cv2.filter2D(temp, -1, left_mask)
		h_i = cv2.filter2D(temp, -1, right_mask)
		chi_sqr_distance = chi_sqr_distance + np.divide(((g_i - h_i)**2),(g_i + h_i + epsilon))
	chi_sqr_distance *= 0.5
	return chi_sqr_distance

def gradient(image, number_bins, disk_masks):
	gradient = image
	for i in range(int(len(disk_masks)/2)):
		gradient_ = chi_sqr_distance(image, number_bins, disk_masks[2*i], disk_masks[2*i+1])
		gradient = np.dstack((gradient, gradient_))
	gradient = np.mean(gradient, axis=2)
	
	return gradient

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [9, 16, 25]
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
	lms_filter_bank = generate_lm_filter(number_orientations, scales, 49)
	draw_lm_filter(lms_filter_bank, number_orientations, len(scales), "LMS")

	scales = [np.sqrt(2)*1, np.sqrt(2)*np.sqrt(2), np.sqrt(2)*2, 2*np.sqrt(2)*np.sqrt(2)]
	lml_filter_bank = generate_lm_filter(number_orientations, scales, 49)
	draw_lm_filter(lml_filter_bank, number_orientations, len(scales), "LML")

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [1, 2, 3, 4]
	theta = 0.25
	Lambda = 1
	psi = 0
	gamma = 1
	number_filters = 10
	gabor_filter_bank = generate_gabor_filters(scales, theta, Lambda, psi, gamma, number_filters)
	draw_gabor_filter(gabor_filter_bank, rows=number_filters, cols=len(scales), filename="gabor")

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	disk_masks = half_disk_masks(scales=[9, 15, 25], number_orientations=8)
	plot_half_disk(disk_masks, 6, 8)

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	images = sorted(glob.glob(\
		'/home/arjun/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/Images/*'))
	texton_images = []
	number_clusters = 64
	for i, image in enumerate(images):
		filename = image.split('/')[-1].split('.')[0]
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		image_ = cv2.imread(image)
		# image_bw = cv2.imread(image, 0)
		texton_map = texton(image_, dog_filter_bank, lms_filter_bank, \
				lml_filter_bank, gabor_filter_bank, number_clusters, filename)
		np.save('Texton_maps/maps_'+str(filename),texton_map)

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		image_bw = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
		brightness_map = brightness(image_bw, filename, number_clusters=16)
		np.save('Brightness_maps/maps_'+str(filename), brightness_map)

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_map = color(image_, filename, number_clusters=16)
		np.save('Color_maps/maps_'+str(filename), color_map)

		# texton_map = np.load('Texton_maps/maps'+str(filename)+'.npy')
		# brightness_map = np.load('Brightness_maps/maps'+str(filename)+'.npy', allow_pickle=True)
		# color_map = np.load('Color_maps/maps'+str(filename)+'.npy', allow_pickle=True)
		# print(texton_map.shape)

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		texton_gradient = gradient(np.array(texton_map), 64, disk_masks)
		print(texton_gradient.shape)
		plt.imsave('./texton_gradient/'+str(filename)+'.png', texton_gradient)

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = gradient(brightness_map, 16, disk_masks)
		plt.imsave('./brightness_gradient/'+str(filename)+'.png', brightness_gradient)

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = gradient(color_map, 16, disk_masks)
		plt.imsave('./color_gradient/'+str(filename)+'.png', color_gradient)

		average_gradient = (texton_gradient + brightness_gradient + color_gradient)/3

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		cannyBaseline = cv2.imread(\
			'/home/arjun/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/'+str(filename)+'.png', 0)
		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobelBaseline = cv2.imread(\
			'/home/arjun/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/'+str(filename)+'.png', 0)
		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		pblite = np.multiply(average_gradient, (0.5*cannyBaseline + 0.5*sobelBaseline))
		plt.imshow(pblite, cmap='gray')
		plt.savefig('./PbLite/'+str(filename)+'.png')
		plt.close()
    
if __name__ == '__main__':
    main()
 
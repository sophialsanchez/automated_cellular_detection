import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy import ndimage as ndi
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu

'''
Function to plot all images, drawing rectangles around segmented cells
Takes as input the labels, the image with noise cleaned (cells_cleaned),
The regions (coordinates to draw boxes around cells), the original color
image (rgb_img), the dilated image (dilated), the ground truth binary cell
image (ground_truth), and the number of cells in the ground truth dataset
(ground_truth_cell_ct)
'''
def plot_all(labels, cells_cleaned, regions, rgb_img, dilated, ground_truth, ground_truth_cell_ct):
	n_cells = len(regions) - 1
	image_label_overlay = label2rgb(labels, image=cells_cleaned)

	fig, axes = plt.subplots(ncols=4, figsize=(9, 4), sharex=True, sharey=True,
	                         subplot_kw={'adjustable': 'box-forced'})
	ax = axes.ravel()

	ax[0].imshow(rgb_img, interpolation='nearest')
	ax[0].set_title('Original Image\n')
	ax[1].imshow(dilated, cmap=plt.cm.gray, interpolation='nearest')
	ax[1].set_title('Segmented Cells\n n = ' + str(n_cells))
	ax[2].imshow(ground_truth, cmap=plt.cm.gray, interpolation='nearest')
	ax[2].set_title('Ground Truth\n n = ' + str(ground_truth_cell_ct))
	ax[3].imshow(image_label_overlay, cmap=plt.cm.gray, interpolation='nearest')
	ax[3].set_title('Separated objects')
	
	for region in regions:
	  	# take regions with large enough areas
	    # draw rectangle around segmented cells
	    minr, minc, maxr, maxc = region.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
	                                  fill=False, edgecolor='red', linewidth=2)
	    ax[3].add_patch(rect)

	for a in ax:
	    a.set_axis_off()

	fig.tight_layout()
	plt.show()

'''
Function to plot the dilated cells. 
Takes in the seed, mask, the dilated image before thresholding,
the original image, and the dilated image after thresholding.
'''
def plot_dilate_cells(seed, mask, dilated_no_thresh, image, dilated):
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))
	yslice = 197

	ax0.plot(mask[yslice], '0.5', label='mask')
	ax0.plot(seed[yslice], 'k', label='seed')
	ax0.plot(dilated_no_thresh[yslice], 'r', label='dilated')
	ax0.set_ylim(-0.2, 2)
	ax0.set_title('Image Slice')
	ax0.set_xticks([])
	ax0.set_xlabel('Pixel')
	ax0.set_ylabel('Intensity')
	ax0.legend()

	ax2.imshow(dilated_no_thresh, vmin=image.min(), vmax=image.max(), cmap='gray')
	ax2.set_title('Dilation Reconstruction')
	ax2.axhline(yslice, color='r', alpha=0.4)
	ax2.axis('off')

	ax1.imshow(image, cmap='gray')
	ax1.set_title('Original Grayscale Image')
	ax1.axhline(yslice, color='r', alpha=0.4)
	ax1.axis('off')

	fig.tight_layout()
	plt.show()

'''
Function to process the cell images using dilation.
Takes in the original image, and a boolean value
describing whether or not to plot the dilated iamge.
Returns the dilated image.
'''
def dilate_cells(image, plot_bool):
	h = 0.4
	seed = image - h
	mask = image

	dilated_no_thresh = reconstruction(seed, mask, method='dilation')
	thresh = threshold_otsu(dilated_no_thresh)
	dilated = dilated_no_thresh > thresh
	dilated = ndi.binary_fill_holes(dilated-1) # Fill in the holes to get more accurate cell objects
	if plot_bool:
		plot_dilate_cells(seed, mask, dilated_no_thresh, image, dilated)

	return(dilated)

'''
Function to clean the cells using masking in order to 
reduce noise in the image dataset. Takes in the 
dilated image and returns the "cleaned" image.
'''
def clean_cells(dilated):
	labeled_cells, _ = ndi.label(dilated)
	sizes = np.bincount(labeled_cells.ravel())
	mask_sizes = sizes > 30
	mask_sizes[0] = 0
	cells_cleaned = mask_sizes[labeled_cells]
	return(cells_cleaned)

'''
Function to analyze the cell image (see steps 1-4 for details).
Takes in a boolean value that describes whether or not to generate plots,
the original cell slide image, the ground truth image (binarized cells),
and the ground truth cell count.
Returns the number of cells in the image.
'''
def analyze_cell_image(plot_bool, curr_cell_image, curr_ground_truth, ground_truth_cell_ct):
	filename = curr_cell_image
	rgb_img = io.imread(filename, plugin='freeimage')
	grey_matrix = color.rgb2grey(io.imread(filename, plugin='freeimage'))
	ground_truth = io.imread(curr_ground_truth, plugin='freeimage')

	# Convert to float: Important for subtraction later which won't work with uint8
	image = img_as_float(grey_matrix)

	# Step 1: Use a Gaussian filter to reduce (high-frequency) noise
	image = gaussian_filter(image, 1)

	# Step 2: Use Dilation reconstruction and Otsu
	# thresholding to isolate the cells and create a 
	# binary image of the cells
	dilated = dilate_cells(image, plot_bool)

	# Step 3: Remove small objects which are noise
	# and not real cells
	cells_cleaned = clean_cells(dilated)

	# Step 4: Perform a distance transform, find the peak local max
	# and use the watershed algorithm to identify 
	distance = ndi.distance_transform_edt(cells_cleaned)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
	                            labels=cells_cleaned)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=cells_cleaned)

	# Clean up the tiny intersections
	regions = regionprops(labels)
	regions = [r for r in regions if r.area > 50]

	n_cells = len(regions) - 1
	print('Number of cells:', str(n_cells))

	# Step 5: Plot images
	if (plot_bool):
		plot_all(labels, cells_cleaned, regions, rgb_img, dilated, ground_truth, ground_truth_cell_ct)

	return(n_cells)

'''
Function to look through all cells in the 
directory of a given type (benign or malignant)
and count the cells in each image.
Returns the total count of cells in all images.
'''
def loop_through_files(cell_type):
	cell_counts = []
	indir = 'BreastCancerCell_cropped_' + cell_type

	ground_truth_filenames = []
	ground_truth_dir = 'BreastCancerCell_ground_truth_cropped'
	for fn in os.listdir(ground_truth_dir):
		if fn != '.DS_Store':
			complete_path = ground_truth_dir + '/' + fn
			ground_truth_filenames.append(complete_path)	

	print(ground_truth_filenames)
	for fn in os.listdir(indir):
		if fn != '.DS_Store':
			curr_cell_image = indir + '/' + fn
			image_num = fn.split('_')[0]
			print(image_num)
			ground_truth_file = ground_truth_filenames[0] # Default to the first image
			ground_truth_cell_ct = ground_truth_filenames[0].split('/')[1].split('_')[3].split('.')[0]
			for img in ground_truth_filenames:
				if img.split('/')[1].split('_')[2] == image_num:
					ground_truth_file = img
					ground_truth_cell_ct = img.split('/')[1].split('_')[3].split('.')[0]
			cell_count = analyze_cell_image(plot_bool, curr_cell_image, ground_truth_file, ground_truth_cell_ct)
			cell_counts.append(cell_count)

	return sum(cell_counts)

'''
Function to plot bar charts. Takes in
the number of cells predicted by the algorithm
and the number of actual cells (i.e. ground truth).
'''
def plot_bar_charts(cells_predicted, cells_actual):
	N = 3 # There are 3 groups - benign, malignant, all
	cells_predicted = cells_predicted

	ind = np.arange(N)  # the x locations for the groups
	width = 0.25       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, cells_predicted, width, color='#2B4A76', alpha=0.7)

	cells_actual = cells_actual
	rects2 = ax.bar(ind + width, cells_actual, width, color='#990000', alpha=0.7)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Cell Count', weight='bold')
	ax.set_xlabel('Cell Type', weight='bold')
	ax.set_ylim([0,cells_actual[2] * 1.15])

	ax.set_title('Cell Count vs. Cell Type', weight='bold')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(('Benign', 'Malignant', 'All'))

	ax.legend((rects1[0], rects2[0]), ('Predicted', 'Actual'))


	def autolabel(rects):
	    """
	    Attach a text label above each bar displaying its height
	    """
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%d' % int(height),
	                ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	plt.show()

'''
Top-level function to analyze all cell images.
Takes in a boolan value which determines 
whether or not to produce plots.
'''
def analyze_cell_images(plot_bool):
	ground_truth_dict = {
		'benign': {
			1: 45,
			2: 49,
			3: 74,
			7: 75,
			8: 64,
			9: 48
		},
		'malignant': {
			4: 42,
			5: 52,
			6: 83,
			10: 32,
			11: 38,
			12: 36
		}
	}

	pred_benign = loop_through_files('benign')
	pred_malignant = loop_through_files('malignant')
	pred_total = pred_benign + pred_malignant
	actual_benign = sum(ground_truth_dict['benign'].values())
	actual_malignant = sum(ground_truth_dict['malignant'].values())
	actual_total = actual_benign + actual_malignant

	print("")
	print("Predicted benign cell count: " + str(pred_benign))
	print("Actual benign cell count: " + str(actual_benign))
	print("")
	print("Predicted malignant cell count: " + str(pred_malignant))
	print("Actual malignant cell count: " + str(actual_malignant))
	print("")
	print("Predicted total cell count: " + str(pred_total))
	print("Actual total cell count: " + str(actual_total))
	plot_bar_charts((pred_benign, pred_malignant, pred_total), (actual_benign, actual_malignant, actual_total))


plot_bool = True
analyze_cell_images(plot_bool)




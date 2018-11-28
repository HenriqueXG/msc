import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import data
import json
import cv2

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

img_name = os.path.join(config['path'], 'debug', 'test_images', 'catdog.jpg')

img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = data.coins()

hist = np.histogram(img, bins=np.arange(0, 256))

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
axes[0].axis('off')
axes[1].plot(hist[1][:-1], hist[0], lw=2)
axes[1].set_title('histogram of gray values')

######################################################################
#
# Thresholding
# ============
#
# A simple way to segment the img is to choose a threshold based on the
# histogram of gray values. Unfortunately, thresholding this image gives a
# binary image that either misses significant parts of the img or merges
# parts of the background with the img:

# fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

# axes[0].imshow(img > 100, cmap=plt.cm.gray, interpolation='nearest')
# axes[0].set_title('img > 100')

# axes[1].imshow(img > 150, cmap=plt.cm.gray, interpolation='nearest')
# axes[1].set_title('img > 150')

# for a in axes:
#     a.axis('off')

# plt.tight_layout()

######################################################################
# Edge-based segmentation
# =======================
#
# Next, we try to delineate the contours of the img using edge-based
# segmentation. To do this, we first get the edges of features using the
# Canny edge-detector.

from skimage.feature import canny

edges = canny(img)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('Canny detector')
# ax.axis('off')

######################################################################
# These contours are then filled using mathematical morphology.

from scipy import ndimage as ndi

fill_img = ndi.binary_fill_holes(edges)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(fill_img, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('filling the holes')
# ax.axis('off')


######################################################################
# Small spurious objects are easily removed by setting a minimum size for
# valid objects.

from skimage import morphology

img_cleaned = morphology.remove_small_objects(fill_img, 21)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(img_cleaned, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('removing small objects')
# ax.axis('off')

######################################################################
# However, this method is not very robust, since contours that are not
# perfectly closed are not filled correctly, as is the case for one unfilled
# coin above.
#
# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

from skimage.filters import sobel

elevation_map = sobel(img)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('elevation map')
# ax.axis('off')

######################################################################
# Next we find markers of the background and the img based on the extreme
# parts of the histogram of gray values.

markers = np.zeros_like(img)
markers[img < 30] = 1
markers[img > 150] = 2

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
# ax.set_title('markers')
# ax.axis('off')

######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:

segmentation = morphology.watershed(elevation_map, markers)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('segmentation')
# ax.axis('off')

######################################################################
# This last method works even better, and the img can be segmented and
# labeled individually.

from skimage.color import label2rgb

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_img, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_img, image=img)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()

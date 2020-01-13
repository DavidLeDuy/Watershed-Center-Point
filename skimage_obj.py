## https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py
## https://blogs.mathworks.com/steve/2013/06/25/homomorphic-filtering-part-1/
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu,gaussian,sobel,laplace,roberts
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb,rgb2gray
from skimage import io as sio

image = sio.imread("img2.png")[50:-50, 50:-50]#data.coins()[50:-50, 50:-50]

plt.imshow(image)
plt.show()
image = rgb2gray(image)
## Homomorphic filtering block to remove light illumination
image = np.log(image) # natural log transform
image = roberts(image) # high pass filter
##
# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))
plt.imshow(bw)
plt.show()

####

###
# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(bw,connectivity=2)
plt.imshow(label_image)
plt.show()
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)
i=0
for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        i+=1
        plt.imsave('segment'+str(i)+'.png',region.image)
        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        print("region detected")
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        #ax.plot(region.centroid,'go')
        # plot centroids
        ax.plot((maxc+minc)/2,(maxr+minr)/2,'ro')

ax.set_axis_off()
plt.tight_layout()
plt.show()
print(label_image.shape)

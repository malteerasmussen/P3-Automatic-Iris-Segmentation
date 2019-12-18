

# print(__doc__)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time


def reduceColor(inputImg, show, n_colors=16):

    # img = cv2.imread(image)
    img = inputImg
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])

    if not isinstance(img, type(None)):
        #Converting to float instead of the default 8 bits interger coding. 
        #Dividing by 255 is important to make sure that plt.show() works well with float data
        #The float data needs to be in the range of 0-1, therefore:
        img = np.array(img, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(img.shape)
        # assert d == 3
        image_array = np.reshape(img, (w * h, d))
        
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        #Running the K-means Clustering
        #The .fit() operation computes the algorithm 
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        # Get labels for all points
        labels = kmeans.predict(image_array)

         # Display all results, alongside original image
        if show is True:
            plt.figure(1)
            plt.clf()
            plt.axis('off')
            plt.title('Quantized image ({} colors, K-Means)'.format(n_colors))
            plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
            plt.show()
        output = recreate_image(kmeans.cluster_centers_, labels, w, h)
        output = output*255
        output = cv2.convertScaleAbs(output)
        #output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # print("output",output.dtype, output)
        return output

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1

    # print("image.dtype",image.dtype)
    return image

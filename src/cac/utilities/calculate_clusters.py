#-----------------------------------------------------
#Import external libraries
import cv2
import numpy
import os

from skimage import io, morphology, measure
from sklearn.cluster import KMeans

#References:
#This code is adopted from Tonechas (2017): https://stackoverflow.com/questions/45043617/count-the-number-of-objects-of-different-colors-in-an-image-in-python
#-----------------------------------------------------


#Function to calculate cluster in a single image
def calculate_colored_clusters(image_array,
                               recognized_number_colors=4,
                               info=False):
    
    #Rescaling image
    rows, cols, channels = image_array.shape
    X = image_array.reshape(rows*cols, channels)

    #Define KMeans
    kmeans = KMeans(n_clusters=recognized_number_colors, random_state=1234567).fit(X)
    labels = kmeans.labels_.reshape(rows, cols)
    
    #Calculate clusters
    collected_colors = []
    collected_counts = []
    for i in numpy.unique(labels):
        blobs = numpy.int_(morphology.binary_opening(labels == i))
        color = numpy.around(kmeans.cluster_centers_[i])
        count = len(numpy.unique(measure.label(blobs))) - 1

        collected_colors.append(color)
        collected_counts.extend([count])
    collected_colors = numpy.asarray(collected_colors).reshape( (len(collected_counts), channels) )

    #Display
    if(info==True):
        for i in range(len(collected_counts)):
            print('The amount of color ( ' + str(int(collected_colors[i, 0])) + ' , ' + str(int(collected_colors[i, 1])) + ' , ' + str(int(collected_colors[i, 2])) + ' ) : ' + str(collected_counts[i]))

    #Return values
    return collected_colors, collected_counts
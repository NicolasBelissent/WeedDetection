from json.tool import main
from libtiff import TIFF
import os
import matplotlib.pyplot as plt
import shapefile as shp
import numpy as np
import pyproj
import rasterio
import math

def extract_coordinates(path_to_shp, x_upper_bound,x_lower_bound, y_upper_bound,y_lower_bound, output_shape):
    sf = shp.Reader(path_to_shp)
    x = []
    y = []
    for shape in sf.shapeRecords():
        x.append([i[0] for i in shape.shape.points[:]][0])
        y.append([i[1] for i in shape.shape.points[:]][0])

    # computing diplacement from bound
    xnorm = (np.array(x) - x_lower_bound) / (x_upper_bound - x_lower_bound)
    ynorm = (np.array(y) - y_lower_bound) / (y_upper_bound - y_lower_bound) 
    
    # scale to number of pixels
    xnorm *= output_shape[0]
    ynorm *= output_shape[1]

    # account for inverted index
    ynorm = np.abs(ynorm-output_shape[1])

    return np.round(xnorm), np.round(ynorm)

def class_matrix(classes, inputsize):
    '''
    Create matrix of weed types and convert labels to numeric
    '''
    # initialize matrix
    matrix = np.zeros((inputsize,inputsize))
    for i in range(inputsize):
        if i in classes[:,0]:
            loc = int(np.where(classes[:,0] == i)[0][0])
            j = classes[:,1][loc]
            class_index = classes[:,2][loc]
            matrix[int(i)][int(j)] = class_index
    
    return matrix

def splitdata(data, classes, splitsize):

    def multiples(m, count):
        val = [i*m for i in range(int(count))]
        return val

    assert classes.shape == data.shape[:2]

    images = []
    labels = []
    #compute batch by batch
    dims = multiples(splitsize, np.round(data.shape[0]/splitsize)+1)
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            try:
                images.append(data[d1:dims[i+1], d2:dims[j+1]])
                labels.append(reformat_labels(np.unique(classes[d1:dims[i+1], d2:dims[j+1]])))
                
            except IndexError:
                continue
                


    return images, labels

def reformat_labels(label):
    new_label = np.zeros(4)
    for lb in label:
        new_label[int(lb)] = 1
    return new_label

if main
    
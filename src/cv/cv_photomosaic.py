import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from collections import Counter
import pdb

#TODO: figure out the bottom of the image edgecase
def cut_mosaic_cv(image, tile_size):
    height, width, channels = image.shape
    hrz_tiles = int(math.floor(width/tile_size))
    vrt_tiles = int(math.floor(height/tile_size))
    hrz_remainder = width - hrz_tiles
    vrt_remainder = height - vrt_tiles
    print("start cutting")
    for x in range(0, hrz_tiles-1):
        for y in range(0, vrt_tiles-1):
            top_l = x*tile_size 
            top_r = x*tile_size + tile_size
            bot_l = y*tile_size 
            bot_r = y*tile_size + tile_size
            tile = image[top_l:bot_l, top_l:top_r]
            color = average_color(image).round()
            #pdb.set_trace()
            #print(tile)
            image[top_l:bot_l, top_l:top_r] = [int(color[0]), int(color[1]), int(color[2])]
    print("cut")
    pdb.set_trace()
    cv2.imshow('mosaic', image)
    cv2.waitKey(0)

#TODO figure out the edgecases
def subdivide(image, row_div, col_div):
    x = 0
    y = 0
    tile_idx = 0
    block_height = math.floor(image.shape[0]/row_div)
    block_width = math.floor(image.shape[1]/col_div)
    # result shape: num of tiles, height, width, channels 
    result = np.zeros((row_div*col_div, block_width, block_height, 3))
    while y < image.shape[0] - block_height:
        y += block_height
        while x < image.shape[1] - block_width:
            x += block_width 
            result[tile_idx] = image[ x:x+block_width, y:y+block_height ]
            tile_idx += 1
    return result


def average_color(image):
    return image.mean(axis=0).mean(axis=0)

def dominant_color(image, clusters=3):
    #convert from bgr to rbg
    #im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #convert image into a list of pixels
    im = image.reshape(image.shape[0]*image.shape[1], 3)
    #create and fit kmeans cluster 
    kmeans = KMeans(n_clusters=clusters)
    labels = kmeans.fit_predict(im)

    lbl_counts = Counter(labels)
    #find the most common (biggest cluster)
    dominant_color = kmeans.cluster_centers_[lbl_counts.most_common(1)[0][0]]
    return list(dominant_color)

def form_mosaic(tiles, tile_height, tile_width, im_shape):
    mosaic = np.zeros(im_shape)
    x = 0
    y = 0
    for tile in tiles:
        mosaic[x:x+tile_width, y:y+tile_height] = np.ones_like(mosaic[x:x+tile_width, y:y+tile_height]) * average_color(tile)
        x += tile_width
        if x > im_shape[1] - tile_width:
            y += tile_height
    return mosaic

def cv_it_up():
    row_div = 10
    col_div = 10
    im = cv2.imread('sample_images/dino.jpg')
    tile_height = math.floor(im.shape[0]/row_div)
    tile_width = math.floor(im.shape[1]/col_div)
    cuts = subdivide(im, row_div, col_div)
    mosaic = form_mosaic(cuts, tile_height, tile_width, im.shape)
    cv2.imshow('mosaic', mosaic)
    cv2.waitKey(0)
    #cut_mosaic_cv(im, 50)

############# TESTING  ##'##############
def test_dominant_color():
    #im = cv2.imread('sample_images/IMG_20180428_221037.jpg')
    #im = cv2.imread('sample_images/IMG_20180304_021058.jpg')
    #im = cv2.imread('sample_images/IMG_20180304_014949.jpg')
    im = cv2.imread('sample_images/IMG_20180428_205935.jpg')
    print(im.shape)
    dsize = (int(im.shape[1] * 0.2), int(im.shape[0] * 0.2))
    print(dsize)
    im = cv2.resize(im, dsize)
    cv2.imshow('original', im)
    cv2.waitKey(0)
    average = dominant_color(im, 1)
    print('average')
    print(average)
    av_im = np.zeros((400,400,3), dtype="uint8")
    av_im[np.where((av_im==[0,0,0]).all(axis=2))] = average
    cv2.imshow('average', av_im)
    cv2.waitKey(0)
    domninant = dominant_color(im)
    print('dominant')
    print(domninant)
    dom_im = np.zeros((400,400,3), dtype="uint8")
    dom_im[np.where((dom_im==[0,0,0]).all(axis=2))] = domninant
    cv2.imshow('dominant', dom_im)
    cv2.waitKey(0)


if __name__ == '__main__':
    #cv_it_up()
    test_dominant_color()
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from collections import Counter
from pathlib import Path
import random
from color_transfer import color_transfer
from colour import delta_E

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
    cv2.imshow('mosaic', image)
    cv2.waitKey(0)

#TODO figure out the edgecases
def subdivide(image, row_div, col_div):
    x = 0
    y = 0
    tile_idx = 0
    block_height = int(image.shape[0]/row_div)
    block_width =  int(image.shape[1]/col_div)
    # result shape: num of tiles, height, width, channels 
    result = np.zeros((row_div*col_div, block_height, block_width, 3), dtype='uint8')
    while y <= image.shape[0] - block_height:
        while x <= image.shape[1] - block_width:
            result[tile_idx] = image[ y:y+block_height , x:x+block_width ]
            x += block_width 
            tile_idx += 1
        y += block_height
        x = 0 #reset x for next row
    print("{} of {} tiles generated".format(tile_idx + 1, len(result)))
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
    dom_color = kmeans.cluster_centers_[lbl_counts.most_common(1)[0][0]]
    return list(dom_color)

def form_mosaic(tiles, tile_height, tile_width, im_shape, transform, dataset=None):
    mosaic = np.zeros(im_shape, dtype="uint8")
    x = 0
    y = 0
    for tile in tiles:
        mosaic[ y:y+tile_height, x:x+tile_width ] = transform(tile, dataset)
        x += tile_width
        if x + tile_width > im_shape[1]:
            y += tile_height
            x = 0
    return mosaic


##### Transforms ######
# transforms take a tile and a dataset and return a modified tile
def color_transfer_transform(tile, choice, dataset=None):
    #average = average_color(tile) #dominant_color(tile, 1)
    #choice = random.choice(dataset)
    insert = np.zeros((tile.shape[0], tile.shape[1], 3), dtype="uint8")
    #color_transfer
    choice = color_transfer(tile, choice, clip=True, preserve_paper=True)
    #TODO figure out why the fuck this is inverted
    min_ht = min(insert.shape[0], choice.shape[0])
    min_wt = min(insert.shape[1], choice.shape[1])
    insert[0:min_ht, 0:min_wt] = choice[0:min_ht, 0:min_wt]
    return insert

def avg_transform(tile, dataset=None):
    average = average_color(tile)
    insert = np.zeros((tile.shape[0], tile.shape[1], 3), dtype="uint8")
    insert[np.where((insert==[0,0,0]).all(axis=2))] = average
    return insert

# TODO: figure out more efficient implementation
def neareast_color_transform(tile, dataset:dict):
    # this is inefficient since we're looping through for each. 
    # need better data structure(custom binary search?)
    # sort: O(N*log(N))
    # search: O(log(N))
    insert = np.zeros((tile.shape[0], tile.shape[1], 3), dtype="uint8")
    closest = None
    max_distance = float("inf")
    target = average_color(tile)
    for color in dataset.keys():
        delta_e = distance_between(target, np.frombuffer(color))
        if delta_e < max_distance:
            max_distance = delta_e
            closest = color
    choice = dataset[closest]
    min_ht = min(insert.shape[0], choice.shape[0])
    min_wt = min(insert.shape[1], choice.shape[1])
    insert[0:min_ht, 0:min_wt] = choice[0:min_ht, 0:min_wt]
    return insert 

def distance_between(color1, color2):
    return np.linalg.norm(color1 - color2)

def scale_down_to_max(image, max_dim):
    #user defined max dimension
    #find longest side
    dim = max(image.shape[0], image.shape[1])

    if dim > max_dim:
        #find scale
        scale = max_dim / dim
        #find new dimensions
        height =  int(scale * image.shape[0])
        width = int(scale * image.shape[1])
        #resize image     # wtf why have a pattern of h, w, c if you're going to makes dimensions be w, h ??????
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    else:
        #don't change
        return image

#TODO set up dataset
# resize and crop
def import_samples(dir, ht, wt, dataset=[]):
    for f in dir.glob('*.jpg'): #just jpg for now
        if f.is_file():
            img = cv2.imread(str(f))
            max_dim = max(ht, wt)
            img = scale_down_to_max(img, max_dim)
            dataset.append(img)
    return dataset

def create_color_map(dataset, estimator):
    color_map = {}
    for img in dataset:
        estimate = estimator(img)
        # TODO: turn the value into a list to support multiple imgs
        # with the same color 
        # TODO: make the np array hashable by converting it to a string (and converting back when needed)
        if estimate.tobytes() not in color_map:
            color_map[estimate.tobytes()] = img
        else:
            print("Warning: found identical color")
    return color_map

def rgb_estimator(img):
    return average_color(img)

def colour_science_estimator():
    pass

def cv_it_up():
    row_div = 40
    col_div = 40
    max_dim = 1080
    im = cv2.imread('sample_images/dino.jpg')
    
    #resize to a smaller image
    #dsize = (int(im.shape[1] * 0.2), int(im.shape[0] * 0.2))
    #im = cv2.resize(im, dsize)
    im = scale_down_to_max(im, max_dim)
    cv2.imshow('downscaled', im)

    tile_height = math.floor(im.shape[0]/row_div)
    tile_width = math.floor(im.shape[1]/col_div)
    print('loading dataset...')
    dataset = import_samples(Path('sample_images'), tile_height, tile_width)
    dataset = import_samples(Path('/home/josh/Pictures/Paris_Trip/Raw'), tile_height, tile_width, dataset)
    dataset = import_samples(Path('/home/josh/Pictures/WV_Trip/Raw'), tile_height, tile_width, dataset)
    color_map = create_color_map(dataset, rgb_estimator)
    print('done')
    cuts = subdivide(im, row_div, col_div)
    transform = lambda tile, dataset: color_transfer_transform( tile, neareast_color_transform(tile, dataset))
    mosaic = form_mosaic(cuts, tile_height, tile_width, im.shape, transform, color_map)
    
    cv2.imwrite('mosaic.png', mosaic)
    #mosaic = form_mosaic(cuts, tile_height, tile_width, im.shape, avg_transform, None)
    #pdb.set_trace()
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
    cv_it_up()
    #test_dominant_color()
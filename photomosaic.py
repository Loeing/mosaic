#!/usr/bin/python3
from PIL import Image
import os
import math
import cv2
import numpy as np
import pdb

def import_samples(sample_path, tile_size, dataset={}):
    for image in os.listdir(sample_path):
        im = Image.open(os.path.join(sample_path, image))
        color = estimate_color(im)
        tile = im.resize((tile_size, tile_size))
        if color not in dataset:
            dataset[color] = tile
        else:
            print('We have an exact duplicate!') #well turn it into lists later
    return dataset

def estimate_color(image):
    average_im = image.resize((1,1))
    color = average_im.getpixel((0,0))
    return color

def find_nearest_color(target, colors):
    #number of ways to do this:
    # - compare distance between colors with colors as 3D points O(N)
    # - bastardized BTree (build: NlogN) search(logN) 
    # - bastardized binary search in sorted list of colors (color as 1 value) (sort: NlogN) (search: logN)
    # 3 channels are weird
    closest = None
    max_distance = float("inf")
    for color in colors:
        if distance_between(color, target) < max_distance:
            max_distance = distance_between(color, target)
            closest = color
    return closest

def distance_between(source, target):
    return math.sqrt((target[0] - source[0]) ** 2 + (target[1] - source[1]) ** 2 + (target[2] - source[2]) ** 2)

#assuming square tiles for now
def cut_mosaic(image, tile_size, dataset):
    width, height = image.size
    hrz_tiles = int(math.floor(width/tile_size))
    vrt_tiles = int(math.floor(height/tile_size))
    for x in range(0, hrz_tiles - 1):
        for y in range(0, vrt_tiles - 1):
            area = (x*tile_size, y*tile_size, x*tile_size+tile_size, y*tile_size+tile_size)
            tile = image.crop(area)
            color = estimate_color(tile)
            #just_color = tile.resize((1,1))#definitely suboptimal
            #image.paste(just_color.resize((tile_size,tile_size)), area)
            neareast_color = find_nearest_color(color,dataset.keys())
            new_tile = dataset[neareast_color]
            image.paste(new_tile, area)
            #print('#{:02x}{:02x}{:02x}'.format(*color))
    return image

def main():
    dataset = import_samples('sample_images', 50)
    dataset = import_samples('/home/josh/Pictures/Paris_Trip/Raw', 50, dataset)
    dataset = import_samples('/home/josh/Pictures/WV_Trip/Raw', 50, dataset)
    #pdb.set_trace()
    im = Image.open('sample_images/dino.jpg')
    image = cut_mosaic(im, 50, dataset)
    image.save('output.png') 

def average_color(image):
    return image.mean(axis=0).mean(axis=0)

def dominant_color(image):
    pass

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
            print(tile)
            image[top_l:bot_l, top_l:top_r] = [int(color[0]), int(color[1]), int(color[2])]
    print("cut")
    pdb.set_trace()
    cv2.imshow('mosaic', image)
    cv2.waitKey(0)


def cv_it_up():
    im = cv2.imread('sample_images/dino.jpg')
    cut_mosaic_cv(im, 50)
    #pdb.set_trace()

if __name__ == '__main__':
    cv_it_up()

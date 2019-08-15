from PIL import Image
import os
import math
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

if __name__ == '__main__':
    main()

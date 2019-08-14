from PIL import Image
import os
import math
import pdb

def import_samples(sample_path):
    for image in os.listdir(sample_path):
        im = Image.open(os.path.join(sample_path, image))

def estimate_color(image):
    average_im = image.resize((1,1))
    color = average_im.getpixel((0,0))
    return color

#assuming square tiles for now
def cut_mosaic(image, tile_size):
    width, height = image.size
    hrz_tiles = int(math.floor(width/tile_size))
    vrt_tiles = int(math.floor(height/tile_size))
    for x in range(0, hrz_tiles - 1):
        for y in range(0, vrt_tiles - 1):
            area = (x*tile_size, y*tile_size, x*tile_size+tile_size, y*tile_size+tile_size)
            tile = image.crop(area)
            color = estimate_color(tile)
            just_color = tile.resize((1,1))#definitely suboptimal
            image.paste(just_color.resize((tile_size,tile_size)), area)
            #print('#{:02x}{:02x}{:02x}'.format(*color))
    image.show()
    image.save('test.png')

    




def main():
    #import_samples('sample_images')
    im = Image.open('sample_images/dino.jpg')
    cut_mosaic(im, 50)

if __name__ == '__main__':
    main()

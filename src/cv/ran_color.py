from color_transfer import color_transfer
import numpy as np
import cv2
import pdb


def main():
    col_src = cv2.imread('sample_images/IMG_20180304_021105.jpg')
    col_target = cv2.imread('sample_images/dino.jpg')
    pdb.set_trace()    
    output = color_transfer(col_src, col_target, clip=True, preserve_paper=True)
    cv2.imshow('src', col_src)
    cv2.imshow('target', col_target)
    cv2.imshow('output', output)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
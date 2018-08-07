import os
import random

import cv2
import numpy as np
from tqdm import tqdm

IMAGES_PATH = "C:/Users/Ron/temp/ayelet_images"
WIDTH_COUNT, HEIGHT_COUNT = 4, 3


def do_the_thing():
    image_paths = search_image_paths()
    image_paths = resample_images(image_paths)
    images = read_images(image_paths)
    images = crop_images(images)
    images = scale_images(images)
    images = pad_images(images)
    new_img = assemble_images(images)
    store_image(new_img)


def search_image_paths():
    return [dir[0] + '/' + f_name
            for dir in os.walk(IMAGES_PATH)
            for f_name in dir[2]]


def read_images(image_paths):
    print('Loading images...')
    return [cv2.imread(img_path) for img_path in tqdm(image_paths)]


def resample_images(images):
    return random.sample(images, WIDTH_COUNT * HEIGHT_COUNT)


def crop_images(images):
    print('Cropping images...')
    min_width = min(img.shape[0] for img in images)
    min_height = min(img.shape[1] for img in images)
    return [img[:min_width, :min_height] for img in images]


def scale_images(images):
    return images


def pad_images(images):
    return images


def assemble_images(images):
    print('Assembling image...')

    first_row = np.concatenate(images[:WIDTH_COUNT], axis=1)
    result = first_row
    for i in range(1, HEIGHT_COUNT):
        next_row = np.concatenate(images[i * WIDTH_COUNT:(i + 1) * WIDTH_COUNT], axis=1)
        result = np.concatenate((result, next_row), axis=0)
    return result


def store_image(new_img):
    print('Storing image...')

    cv2.imshow('image', new_img)
    cv2.imwrite('result.jpg', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do_the_thing()

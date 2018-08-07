import os
import random
from math import sqrt

import cv2
import numpy as np

IMAGES_PATH = "C:/Users/Ron/Dropbox/Code/py/home_collage/sample_images"


def do_the_thing():
    images = get_all_images()
    images = sample_images(images)
    images = crop_images(images)
    images = pad_images(images)
    new_img = assemble_images(images)
    store_image(new_img)


def get_all_images():
    return [cv2.imread(dir[0] + '/' + f_name) for dir in os.walk(IMAGES_PATH) for f_name in dir[2]]


def sample_images(images):
    n = grid_size(len(images))
    return random.sample(images, n ** 2)


def crop_images(images):
    min_width = min(img.shape[0] for img in images)
    min_height = min(img.shape[1] for img in images)
    return [img[:min_width, :min_height] for img in images]


def pad_images(images):
    return images  # TODO - implement!


def assemble_images(images):
    num_images = len(images)
    n = grid_size(num_images)
    result = np.concatenate(images[:n], axis=1)

    for i in range(1, n):
        new_row = np.concatenate(images[i * n:(i + 1) * n], axis=1)
        result = np.concatenate((result, new_row), axis=0)
    return result


def store_image(new_img):
    cv2.imshow('image', new_img)
    cv2.imwrite('result.jpg', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grid_size(num_images):
    return int(sqrt(num_images))


if __name__ == '__main__':
    do_the_thing()

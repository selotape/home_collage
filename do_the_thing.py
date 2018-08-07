import os

import cv2
import numpy as np

IMAGES_PATH = "C:/Users/Ron/Dropbox/Code/py/home_collage/sample_images"


def get_all_images():
    return [cv2.imread(dir[0] + '/' + f_name) for dir in os.walk(IMAGES_PATH) for f_name in dir[2]]


def crop_images(images):
    min_width = min(img.shape[0] for img in images)
    min_height = min(img.shape[1] for img in images)
    return [img[:min_width, :min_height] for img in images]


def pad_images(images):
    return images  # TODO - implement!


def assemble_images(images):
    return np.concatenate(images, axis=1)


if __name__ == '__main__':
    images = get_all_images()
    images = crop_images(images)
    images = pad_images(images)
    new_img = assemble_images(images)

    cv2.imshow('image', new_img)
    cv2.imwrite('result.jpg', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

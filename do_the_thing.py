from __future__ import division

import os
import random
from datetime import datetime
from functools import partial

import cv2
import numpy as np

from util import height, width

### CONF ###

IMAGES_PATH = "c:/users/ron/temp/ayelet_images"
# IMAGES_PATH = "C:/Users/Ron/Dropbox/Code/py/home_collage/sample_images"
OUTPUT_PATH = "C:/Users/Ron/temp/result.jpg"
WIDTH_IN_PIXELS, HEIGHT_IN_PIXELS = 70000, 30000
IMG_HEIGHT_IN_PIXELS = 1440
IMAGE_PAD_IN_PIXELS, COLLAGE_PAD_IN_PIXELS = 24, 144

### CONSTS ###
WHITE = [255, 255, 255]


def do_the_thing():
    print('Start time: %s' % str(datetime.now()))
    print_metrics()
    image_paths = search_image_paths()
    images = read_images(image_paths)
    images = filter_images(images)
    images = rotate_images(images)
    images = scale_images(images)
    images = pad_images(images)
    new_img = assemble_collage(images)
    new_img = pad_collage(new_img)
    store_image(new_img)
    print('End time: %s' % str(datetime.now()))


def print_metrics():
    print('===============================')
    print('Expected number of rows: %d' % (HEIGHT_IN_PIXELS // IMG_HEIGHT_IN_PIXELS))
    print('Expected images per row: %d' % (WIDTH_IN_PIXELS // IMG_HEIGHT_IN_PIXELS))
    print('Expected total images: %d' % (HEIGHT_IN_PIXELS * WIDTH_IN_PIXELS // IMG_HEIGHT_IN_PIXELS ** 2))
    print('===============================\n')


def search_image_paths():
    image_paths = [dir[0] + '/' + f_name for dir in os.walk(IMAGES_PATH) for f_name in dir[2] if f_name.endswith(('jpg', 'JPG'))]
    random.shuffle(image_paths)
    return image_paths


def read_images(image_paths):
    for i, img_path in enumerate(image_paths, start=1):
        print("Reading %dth image ('%s')..." % (i, img_path))
        yield cv2.imread(img_path)


def filter_images(images):
    return images


def rotate_images(images):
    return images


def scale_images(images):
    for i, image in enumerate(images, start=1):
        yield scale_image(image)


def scale_image(image):
    scale = height(image) / IMG_HEIGHT_IN_PIXELS
    scaled_size = (int(width(image) / scale), IMG_HEIGHT_IN_PIXELS)
    return cv2.resize(src=image, dsize=scaled_size, interpolation=cv2.INTER_AREA)


def pad_images(images):
    pad = partial(cv2.copyMakeBorder,
                  top=IMAGE_PAD_IN_PIXELS, bottom=IMAGE_PAD_IN_PIXELS, left=IMAGE_PAD_IN_PIXELS,
                  right=IMAGE_PAD_IN_PIXELS,
                  borderType=cv2.BORDER_CONSTANT, value=WHITE)
    return (pad(img) for img in images)


def assemble_collage(images):
    images = iter(images)
    num_rows = HEIGHT_IN_PIXELS // IMG_HEIGHT_IN_PIXELS
    rows = [assemble_row(images) for _ in range(num_rows)]
    return np.concatenate(rows, axis=0)


def assemble_row(images):
    row_images = choose_row_images(images)
    print('Images in row: %d' % len(row_images))
    complete_missing_columns(row_images)
    return np.concatenate(row_images, axis=1)


def choose_row_images(images):
    row_images = []
    next_img = next(images)
    while sum(width(img) for img in row_images) + width(next_img) <= WIDTH_IN_PIXELS:
        row_images.append(next_img)
        next_img = next(images)
    return row_images


def complete_missing_columns(row_images):
    # type: (list) -> None
    missing_columns = WIDTH_IN_PIXELS - sum(width(img) for img in row_images)
    num_images = len(row_images)
    while missing_columns >= num_images - 1:
        for i in range(num_images - 1):
            right_pad_row_image(i, row_images)
            missing_columns -= 1

    while missing_columns > 0:
        rand_ix = random.randrange(num_images - 1)
        right_pad_row_image(rand_ix, row_images)
        missing_columns -= 1

    final_row_width = sum(width(img) for img in row_images)
    assert final_row_width == WIDTH_IN_PIXELS, "Row doesn't reach target width. target:%d. actual:%d" % (WIDTH_IN_PIXELS, final_row_width)


def right_pad_row_image(img_ix, row_images):
    img = row_images.pop(img_ix)
    img = cv2.copyMakeBorder(src=img, top=0, bottom=0, left=0, right=1,
                             borderType=cv2.BORDER_CONSTANT, value=WHITE)
    row_images.insert(img_ix, img)


def pad_collage(collage):
    return cv2.copyMakeBorder(collage,
                              top=COLLAGE_PAD_IN_PIXELS, bottom=COLLAGE_PAD_IN_PIXELS, left=COLLAGE_PAD_IN_PIXELS, right=COLLAGE_PAD_IN_PIXELS,
                              borderType=cv2.BORDER_CONSTANT, value=WHITE)


def store_image(new_img):
    print("Storing image in '%s'..." % OUTPUT_PATH)
    cv2.imwrite(OUTPUT_PATH, new_img)
    os.system("start " + OUTPUT_PATH)


if __name__ == '__main__':
    do_the_thing()

import os
import random
from functools import partial

import cv2
import numpy as np

### CONF ###
IMAGES_PATH = "C:/Users/Ron/temp/ayelet_images"
# IMAGES_PATH = "C:/Users/Ron/Dropbox/Code/py/home_collage/sample_images"
OUTPUT_PATH = "C:/Users/Ron/temp/result.jpg"
WIDTH_IN_PIXELS, HEIGHT_IN_PIXELS = 4000, 2500
IMG_HEIGHT_IN_PIXELS = 120
IMAGE_PAD_IN_PIXELS, COLLAGE_PAD_IN_PIXELS = 2, 5

### CONSTS ###
WHITE = [255, 255, 255]


def print_metrics():
    print('Expected num rows: %d' % (HEIGHT_IN_PIXELS // IMG_HEIGHT_IN_PIXELS))
    print('Expected imgs per rows: %d' % (WIDTH_IN_PIXELS // IMG_HEIGHT_IN_PIXELS))
    print('Expected total images: %d' % (HEIGHT_IN_PIXELS * WIDTH_IN_PIXELS // IMG_HEIGHT_IN_PIXELS ** 2))


def do_the_thing():
    print_metrics()
    image_paths = search_image_paths()
    images = read_images(image_paths)
    images = scale_images(images)
    images = pad_images(images)
    new_img = assemble_collage(images)
    new_img = pad_collage(new_img)
    store_image(new_img)


def search_image_paths():
    image_paths = [dir[0] + '/' + f_name for dir in os.walk(IMAGES_PATH) for f_name in dir[2]]
    random.shuffle(image_paths)
    return image_paths


def read_images(image_paths):
    for i, img_path in enumerate(image_paths, start=1):
        print("Reading %dth image ('%s')..." % (i, img_path))
        yield cv2.imread(img_path)


def pad_collage(collage):
    return cv2.copyMakeBorder(collage,
                              top=IMAGE_PAD_IN_PIXELS, bottom=IMAGE_PAD_IN_PIXELS, left=IMAGE_PAD_IN_PIXELS, right=IMAGE_PAD_IN_PIXELS,
                              borderType=cv2.BORDER_CONSTANT, value=WHITE)


def scale_images(images):
    for i, image in enumerate(images, start=1):
        print("Scaling %dth image..." % i)
        yield scale_image(image)


def scale_image(image):
    scale = height(image) // IMG_HEIGHT_IN_PIXELS
    scaled_size = (width(image) // scale, IMG_HEIGHT_IN_PIXELS)
    return cv2.resize(src=image, dsize=scaled_size, interpolation=cv2.INTER_AREA)


def height(img):
    return img.shape[0]


def width(img):
    return img.shape[1]


def pad_images(images):
    pad = partial(cv2.copyMakeBorder,
                  top=IMAGE_PAD_IN_PIXELS, bottom=IMAGE_PAD_IN_PIXELS, left=IMAGE_PAD_IN_PIXELS,
                  right=IMAGE_PAD_IN_PIXELS,
                  borderType=cv2.BORDER_CONSTANT, value=WHITE)
    return (pad(img) for img in images)


def assemble_collage(images):
    images = iter(images)
    first_row = assemble_row(images)
    collage = first_row
    num_rows = HEIGHT_IN_PIXELS // IMG_HEIGHT_IN_PIXELS
    for _ in range(num_rows - 1):
        next_row = assemble_row(images)
        collage = np.concatenate((collage, next_row), axis=0)
    return collage


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
        for i in range(num_images):
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


def store_image(new_img):
    print('Storing image...')

    cv2.imshow('image', new_img)
    cv2.imwrite(OUTPUT_PATH, new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do_the_thing()

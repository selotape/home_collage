import os
import random
from functools import partial

import cv2
from tqdm import tqdm

# IMAGES_PATH = "C:/Users/Ron/temp/ayelet_images"
IMAGES_PATH = "C:/Users/Ron/Dropbox/Code/py/home_collage/sample_images"
OUTPUT_PATH = "C:/Users/Ron/temp/result.jpg"
WIDTH_IN_PIXELS, HEIGHT_IN_PIXELS = 6780, 3012
IMG_HEIGHT_IN_PIXELS = 140
INNER_PAD_IN_PIXELS, OUTER_PAD_IN_PIXELS = 2, 5
PAD_COLOR = 'WHITE'


def do_the_thing():
    image_paths = search_image_paths()
    images = read_images(image_paths)
    images = scale_images(images)
    images = pad_images(images)
    new_img = assemble_image(images)
    new_img = pad_image(new_img)
    store_image(new_img)


def search_image_paths():
    image_paths = [dir[0] + '/' + f_name for dir in os.walk(IMAGES_PATH) for f_name in dir[2]]
    random.shuffle(image_paths)
    return image_paths


def read_images(image_paths):
    return (cv2.imread(img_path) for img_path in tqdm(image_paths))


def pad_image(new_img):
    return new_img


def scale_images(images):
    for image in images:
        yield scale_image(image)


def scale_image(image):
    scale = height(image) // IMG_HEIGHT_IN_PIXELS
    scaled_size = (width(image) // scale, IMG_HEIGHT_IN_PIXELS)
    return cv2.resize(src=image, dsize=scaled_size, interpolation=cv2.INTER_AREA)


def height(img):
    return img.shape[0]


def width(img):
    return img.shape[1]


WHITE = [255, 255, 255]


def pad_images(images):
    pad = partial(cv2.copyMakeBorder,
                  top=INNER_PAD_IN_PIXELS, bottom=INNER_PAD_IN_PIXELS, left=INNER_PAD_IN_PIXELS, right=INNER_PAD_IN_PIXELS,
                  borderType=cv2.BORDER_CONSTANT, value=WHITE)
    return (pad(img) for img in images)


def assemble_image(images):
    # print('Assembling image...')
    #
    # first_row = np.concatenate(images[:WIDTH_COUNT], axis=1)
    # result = first_row
    # for i in range(1, HEIGHT_COUNT):
    #     next_row = np.concatenate(images[i * WIDTH_COUNT:(i + 1) * WIDTH_COUNT], axis=1)
    #     result = np.concatenate((result, next_row), axis=0)
    return next(images)


def store_image(new_img):
    print('Storing image...')

    cv2.imshow('image', new_img)
    cv2.imwrite(OUTPUT_PATH, new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do_the_thing()

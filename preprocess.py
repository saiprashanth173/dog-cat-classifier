import cv2


def resize_images(images, dims):
    return [cv2.resize(image, dims) for image in images]

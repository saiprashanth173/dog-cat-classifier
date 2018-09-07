import cv2


def conv_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, dims):
    return cv2.resize(image, dims)


def preprocess(images, preprocess_functions):
    processed_images = []
    for image in images:
        processed = image
        for function in preprocess_functions:
            processed = function(processed)
        processed_images.append(processed)
    return processed_images

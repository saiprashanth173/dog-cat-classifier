import os
import cv2
from tqdm import tqdm


def label_from_name(img):
    return img.split('.')[-3]


def load_images_with_labels(path):
    X = []
    y = []
    for img in tqdm(os.listdir(path)):
        label = label_from_name(img)
        path = os.path.join()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        X.append(img)
        y.append(label)
    return X, y

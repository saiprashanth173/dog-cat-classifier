import os
import cv2
from tqdm import tqdm
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


def label_from_name(img):
    return [1, 0] if img.split('.')[0] == "cat" else [0, 1]


def load_images_with_labels(path):
    X = []
    y = []
    files = os.listdir(path)
    shuffle(files)
    for img in tqdm(files):
        label = label_from_name(img)
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        X.append(image)
        y.append(label)
    return X, y


def plot_predictions(images, predictions, plot_file):
    fig = plt.figure()
    for index, (prediction, image) in enumerate(zip(predictions, images)):
        y = fig.add_subplot(3, 4, index + 1)
        if np.argmax(prediction) == 1:
            str_label = 'Dog'
            confidence = prediction[1] * 100
        else:
            str_label = 'Cat'
            confidence = prediction[0] * 100
        y.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("{}({:.2f}%)".format(str_label, confidence))
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.savefig(plot_file, dpi=fig.dpi)

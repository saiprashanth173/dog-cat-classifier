from config import IMG_SIZE, MODEL_NAME, LR, DEFAULT_MODEL_DIRECTORY, DEFAULT_TEST_DIRECTORY
from models import deepconvnet
from preprocess import resize_image, conv_gray_scale, preprocess
from utils import load_images_with_labels, plot_predictions
import numpy as np

import time


def test(model_name, plot_file=None, test_directory=DEFAULT_TEST_DIRECTORY, model_directory=DEFAULT_MODEL_DIRECTORY):
    model = deepconvnet(model_name, LR, (IMG_SIZE, IMG_SIZE), model_directory=model_directory)
    X_orig, y_test = load_images_with_labels(test_directory)
    X = preprocess(X_orig, [conv_gray_scale, lambda x: resize_image(x, (IMG_SIZE, IMG_SIZE))])
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    start = time.time()
    predictions = model.predict(X)
    print("\nTime taken to predict outcomes: {} secs \n".format(time.time() - start))

    if plot_file:
        plot_predictions(X_orig, predictions, plot_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default=MODEL_NAME, help="Name of the model to be trained")
    parser.add_argument("-p", "--model_path", default=DEFAULT_MODEL_DIRECTORY, help="Path to the model")
    parser.add_argument("-t", "--test_directory", default=DEFAULT_TEST_DIRECTORY, help="Path to train directory")
    parser.add_argument("--plot_file", help="Predictions will be plotted and image will be saved in the file specified")

    args = parser.parse_args()

    start = time.time()
    test(args.model_name, plot_file=args.plot_file, test_directory=args.test_directory, model_directory=args.model_path)
    print("Total time for execution: {} secs\n".format(time.time() - start))

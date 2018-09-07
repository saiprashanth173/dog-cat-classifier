import os
import time

from config import MODEL_NAME, IMG_SIZE, LR, DEFAULT_MODEL_DIRECTORY, DEFAULT_TRAIN_DIRECTORY, DEFAULT_EPOCH
from models import deepconvnet
from utils import load_images_with_labels
from preprocess import resize_image, preprocess, conv_gray_scale
import numpy as np
from sklearn.model_selection import train_test_split


def train(model_name, train_directory=DEFAULT_TRAIN_DIRECTORY, epoch=DEFAULT_EPOCH, learning_rate=LR,
          model_save_path=DEFAULT_MODEL_DIRECTORY):
    model = deepconvnet(model_name, learning_rate, (IMG_SIZE, IMG_SIZE))
    X, y = load_images_with_labels(train_directory)
    X = preprocess(X, [conv_gray_scale, lambda x: resize_image(x, (IMG_SIZE, IMG_SIZE))])
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(
        {'input': X_train},
        {'targets': y_train},
        validation_set=({"input": X_validate}, {"targets": y_validate}),
        n_epoch=epoch,
        snapshot_step=500,
        show_metric=True,
        run_id=model_name
    )
    model.save(os.path.join(model_save_path, model_name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default=MODEL_NAME, help="Name of the model to be trained")
    parser.add_argument("-e", "--epoch", type=int, default=DEFAULT_EPOCH,
                        help="Number of epochs to train the model - default (3)")
    parser.add_argument("-l", "--learning_rate", type=float, default=LR,
                        help="Learning Rate for training the model (float) - default {}".format(LR))
    parser.add_argument("-p", "--model_path", default=DEFAULT_MODEL_DIRECTORY, help="Path to the model")
    parser.add_argument("-t", "--train_directory", default=DEFAULT_TRAIN_DIRECTORY, help="Path to train directory")

    args = parser.parse_args()
    start = time.time()
    train(args.model_name, epoch=args.epoch, learning_rate=args.learning_rate, train_directory=args.train_directory,
          model_save_path=args.model_path)
    print("Total time for execution: {} secs\n".format(time.time() - start))

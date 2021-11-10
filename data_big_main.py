from big_dataset import load_mnist, plot_images

from test import *
from train_and_dump import *

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_mnist()  # Call the data reading function of another python file
    # train(X_train, Y_train)
    test_eval(X_train, X_test, Y_test)


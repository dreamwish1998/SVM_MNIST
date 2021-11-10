import os
import os.path
import urllib.request
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2

if not os.path.exists('mnist'):  # Create the mnist data set (that is, create it if you don't have it)
    os.mkdir("mnist")  # Create mnist data set


def download_and_gzip(name):  # Download the data set
    if not os.path.exists(name + '.gz'):
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/' + name + '.gz', name + '.gz')  # Get compressed data through this statement
    if not os.path.exists(name):
        with gzip.open(name + '.gz', "rb") as f_in, open(name, 'wb') as f_out:  # Obtain the corresponding file without suffix
            shutil.copyfileobj(f_in, f_out)  # Put it in the current setting directory


download_and_gzip("mnist/train-images-idx3-ubyte")  # Call the function to get and store the data
download_and_gzip('mnist/train-labels-idx1-ubyte')  # Same as above
download_and_gzip('mnist/t10k-images-idx3-ubyte')  # Same as above
download_and_gzip("mnist/t10k-labels-idx1-ubyte")  # Same as above


def load_mnist():
    loaded = np.fromfile("mnist/train-images-idx3-ubyte", dtype='uint8')  # Convert training data to numpy type
    train_x = loaded[16:].reshape(60000, 28, 28)  # Convert the data from 784 to 28*28
    loaded = np.fromfile("mnist/t10k-images-idx3-ubyte", dtype='uint8')  # Same as above but becomes a test set
    test_x = loaded[16:].reshape(10000, 28, 28)  # Same as above
    loaded = np.fromfile('mnist/train-labels-idx1-ubyte', dtype='uint8')  # Obtain the training data label and convert it into numpy type
    train_y = loaded[8:].reshape(60000)  # Same as above
    loaded = np.fromfile("mnist/t10k-labels-idx1-ubyte", dtype='uint8')  # Test set label to numpy
    test_y = loaded[8:].reshape(10000)  # Same as above
    return train_x, train_y, test_x, test_y

def plot_images(images, row, col):  # Draw a graph in the specified data set according to the range of rows and columns
    show_image = np.vstack(np.split(np.hstack(images[:col * row]), row, axis=1))  # Get the photo gallery
    plt.imshow(show_image, cmap='binary')  # Show pictures
    plt.axis("off")  # Turn off axis scale
    plt.show()  # Display canvas

# row, col = 1, 1
# train_x, train_y, test_x, test_y = load_mnist()
# plot_images(train_x, row, col)
# cv2.imwrite("big_tesult_0.jpg", train_x[1])

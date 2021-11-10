from big_dataset import load_mnist, plot_images
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def confusion_plot(confusion):
    # Draw confusion matrix
    for first_index in range(len(confusion)):  # Which row
        for second_index in range(len(confusion[first_index])):  # which column
            plt.text(first_index, second_index, confusion[first_index][second_index])  # plot point label
            plt.title('confusion matrix')
            x_major_locator = MultipleLocator(1)  # Set the abscissa scale interval
            y_major_locator = MultipleLocator(1)  # Set the ordinate scale interval
            ax = plt.gca()  # Get the current Axes object
            ax.xaxis.set_major_locator(x_major_locator)  # x
            ax.yaxis.set_major_locator(y_major_locator)  # y
            plt.ylabel('Predict')  # set label
            plt.xlabel('Truth')  # Same as above
            plt.imshow(confusion, interpolation='nearest', cmap='Reds')  # show confusion
    plt.show()

def plot_result(X_test, Y_test, Y_predict):
    # No matching correct index was found
    false_list = []  # Define an empty list to store the prediction error label
    for m in range(10000):  # Cycle 10000 test set
        if Y_test[m] != Y_predict[m]:
            false_list.append(m)  # Save the label predicted to be misclassified according to the conditions

    # plot
    # Show classification results
    for i in range(7):
        randint = i + 200  # Interval setting
        if Y_test[i] == Y_predict[i]:  # Show test samples that are correctly classified
            plt.subplot(2, 4, i + 1)  # 2 rows 4 columns
            plt.title('Truth: {0}\nPredict: {1}'.format(Y_test[randint], Y_predict[randint]),
                      color='blue')  # Set title
            plt.imshow(X_test[randint], cmap='gray')  # Show pictures
            plt.xticks([])  # Do not scale on x
            plt.yticks([])  # Do not scale on y

    plt.subplot(2, 4, 8)  # Wrong sample display
    plt.title('Truth: {0}\nPredict: {1}'.format(Y_test[false_list[0]], Y_predict[false_list[0]]), color='red')  # same as above
    plt.imshow(X_test[false_list[0]], cmap='gray')  # same as above
    plt.xticks([])  # same as above
    plt.yticks([])  # same as above
    plt.show()


def test_eval(X_train, X_test, Y_test):
    digits = load_digits()  # Call the load_digits() method
    train_x = X_train
    test_x = X_test
    y_test = Y_test
    # X_train, Y_train, X_test, Y_test = load_mnist()  # Call the data reading function of another python file
    X_train = X_train.reshape(-1, 28 * 28).astype("float") / 255  # Reshape the train set data
    X_test = X_test.reshape(-1, 28*28).astype("float")/255  # Reshape the test set data

    ss = StandardScaler()  # Call standardized function
    # it is an instance method and must be called by the instance
    X_train = ss.fit_transform(X_train)  # Normalize the training set
    X_test = ss.transform(X_test)  # Normalize the test set

    clf = joblib.load('models/clf.pkl')  # Import the previously saved model
    Y_predict = clf.predict(X_test)  # Use the model to make classification predictions on the test set

    # result analysis tool
    print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str)))

    # confusion matrix
    confusion = confusion_matrix(Y_test, Y_predict)
    confusion_plot(confusion)
    plot_result(test_x, y_test, Y_predict)



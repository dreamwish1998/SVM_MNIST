from big_dataset import load_mnist, plot_images
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

def train(X_train, Y_train):
    digits = load_digits()  # Call the load_digits() method
    # X_train, Y_train, X_test, Y_test = load_mnist()  # Call the data reading function of another python file
    X_train = X_train.reshape(-1, 28*28).astype("float")/255  # Reshape the training data

    ss = StandardScaler()  # Call standardized function

    # fit is an instance method and must be called by the instance
    X_train = ss.fit_transform(X_train)  # Normalize the training set

    svc = SVC(kernel='rbf')  # Use the SVC function to define the SVM model and set the kernel function at the same time
    svc.fit(X_train, Y_train)  # Classifier training

    joblib.dump(svc, 'models/clf_rbf.pkl')  # Save the trained model (different according to the definition of the kernel function)


import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from joblib import dump, load
from config import SVM_MODEL_FILE

def train_svm(feature_vectors, labels, model_file=SVM_MODEL_FILE):
    X = np.array(feature_vectors)
    y = np.array(labels)

    kf = KFold(n_splits=5)
    best_accuracy = 0
    best_model = None
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        print(f'Accuracy for fold: {accuracy * 100:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
    
    dump(best_model, model_file)
    print(f'Model saved to {model_file} with accuracy: {best_accuracy * 100:.2f}%')

def load_svm_model(model_file=SVM_MODEL_FILE):
    return load(model_file)

def predict_svm(model, feature_vectors):
    return model.predict(feature_vectors)

def predict_svm_proba(model, feature_vectors):
    return model.predict_proba(feature_vectors)

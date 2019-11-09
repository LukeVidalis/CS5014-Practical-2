import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib.font_manager import FontProperties

def read_data(id):
    binary_file_path = "binary\\"
    multiclass_file_path = "multiclass\\"

    if id == 1:
        file_path = binary_file_path
        key = {'book': 0, 'plastic case': 1}
    elif id == 2:
        file_path = multiclass_file_path
        key = {'air': 0, 'book': 1, 'hand': 2, 'knife': 3, 'plastic case': 4}

    X = pd.read_csv(file_path+"X.csv", header=None)

    XToClassify = pd.read_csv(file_path+"XToClassify.csv", header=None)
    y = pd.read_csv(file_path+"y.csv", header=None)

    return X, y, XToClassify, key


def split_data(X, y):
    X_train, X_testing, y_train, y_testing = train_test_split(X, y, test_size=0.20, shuffle=True,
                                                              random_state=40)

    return X_train, y_train, X_testing, y_testing


def scale_data(X,y, X2):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scaler.fit(X2)
    XToClassify = scaler.transform(X2)
    scaler.fit(y)
    y = scaler.transform(y)
    return X, y, XToClassify


def plot_graphs(X, y, key):
    X = X.values
    y = y.values
    X_mean = X[:, 0:256]
    X_min = X[:, 256:2 * 256]
    X_max = X[:, 2 * 256:3 * 256]
    bound = list(range(0, 256))

    num_of_classes = len(np.unique(y))

    sections = [X_mean, X_min, X_max]
    titles = ["Mean", "Min", "Max"]
    plt.tight_layout()
    fontP = FontProperties()
    fontP.set_size('small')

    index = 0
    for X in sections:

        plt.subplot(3, 1, index+1).set_title(titles[index] + " Section")
        for c in range(num_of_classes):
            values = np.mean(X[np.where(y[:, 0] == c)], axis=0)
            plt.scatter(bound, values)

        plt.legend(key, prop=fontP)
        index += 1
    plt.tight_layout(h_pad=0.30)
    plt.show()


def logistic_regression(X_train, y_train, X_test, y_test, class_list, XToClassify):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    if len(class_list) == 2:
        title = "binary"
    else:
        title = "multiclass"
    print("Logistic Regression:")
    print("")
    classified_y = model.predict(XToClassify)
    np.savetxt("output\\logistic_regression_"+title+".csv", classified_y, delimiter=",")
    pt1, pt2 = calc_scores(model, X_train, y_train, X_test, y_test, class_list)
    pt1.savefig('confusion_matrices\\logistic_regression_train_'+title+'.png')
    pt2.savefig('confusion_matrices\\logistic_regression_test_'+title+'.png')

    return model, classified_y


def linear_svc(X_train, y_train, X_test, y_test, class_list, XToClassify):
    model = LinearSVC()
    model.fit(X_train, y_train)
    if len(class_list) == 2:
        title = "binary"
    else:
        title = "multiclass"
    print("SVC:")
    print("")
    classified_y = model.predict(XToClassify)
    np.savetxt("output\\linear_svc_" + title + ".csv", classified_y, delimiter=",")
    pt1, pt2 = calc_scores(model, X_train, y_train, X_test, y_test, class_list)
    pt1.savefig('confusion_matrices\\linear_svc_train_' + title + '.png')
    pt2.savefig('confusion_matrices\\linear_svc_test_' + title + '.png')
    return model


def decision_tree(X_train, y_train, X_test, y_test, class_list, XToClassify):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    if len(class_list) == 2:
        title = "binary"
    else:
        title = "multiclass"
    print("Decision Tree")
    print("")
    classified_y = model.predict(XToClassify)
    np.savetxt("output\\decision_tree_" + title + ".csv", classified_y, delimiter=",")
    pt1, pt2 = calc_scores(model, X_train, y_train, X_test, y_test, class_list)
    pt1.savefig('confusion_matrices\\decision_tree_train_' + title + '.png')
    pt2.savefig('confusion_matrices\\decision_tree_test_' + title + '.png')

    return model


def calc_scores(model, X_train, y_train, X_test, y_test, key):
    if int(len(key)) == 2:
        avg_mode = "binary"
    else:
        avg_mode = 'macro'

    y_train_pred = model.predict(X_train)

    pt1 = plot_confusion_matrix(y_train, y_train_pred, key)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print("Cross validation scores: " + str(scores))
    print("Cross validation accuracy: " + str(np.mean(scores)))
    error = cross_val_score(model, X_test, y_test, cv=5,
                            scoring="neg_mean_squared_error").mean()
    print("SME Testing Set:", -error)
    y_test_pred = model.predict(X_test)
    pt2 = plot_confusion_matrix(y_test, y_test_pred, key)

    print("Accuracy scores: " + str(accuracy_score(y_test, y_test_pred)))
    print("Precision scores: " + str(precision_score(y_test, y_test_pred, average=avg_mode)))
    print("")
    return pt1, pt2

# Method for plotting confusion matrix.
# Taken from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.show()
    return plt


def create_dir():
    # Create directory
    dirName = 'output'

    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    dirName = 'confusion_matrices'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


if __name__ == "__main__":
    create_dir();
    print("Binary Classifier:")
    print("")

    X, y, XToClassify, key = read_data(1)
    plot_graphs(X, y, key)
    # X,y, XToClassify = scale_data(X, y, XToClassify)
    X_train, y_train, X_testing, y_testing = split_data(X, y)

    model1a, y_classified = logistic_regression(X_train, y_train, X_testing, y_testing, key, XToClassify)
    print(y_classified)
    model2a = linear_svc(X_train, y_train, X_testing, y_testing, key, XToClassify)
    model3a = decision_tree(X_train, y_train, X_testing, y_testing, key, XToClassify)
    print("Multiclass Classifier:")
    print("")

    X, y, XToClassify, key = read_data(2)
    plot_graphs(X, y, key)
    X_train, y_train, X_testing, y_testing = split_data(X, y)

    model1b = logistic_regression(X_train, y_train, X_testing, y_testing, key, XToClassify)
    model2b = linear_svc(X_train, y_train, X_testing, y_testing, key, XToClassify)
    model3b = decision_tree(X_train, y_train, X_testing, y_testing, key, XToClassify)

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

run_summary = False
run_visualisation = False
run_evaluation = True
run_prediction = False

# Load dataset
filename = "iris.data.txt"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(filename, names=names)
input("Press Enter to continue...")

if run_summary:
    # Print shape
    print("Dataset shape:")
    print(dataset.shape)
    input("Press Enter to continue...")

    # Peek at data set
    print("First 20 rows of dataset:")
    print(dataset.head(20))
    input("Press Enter to continue...")

    # Statistical descriptions (mean, std, etc.)
    print("Statistical desctiption of dataset")
    print(dataset.describe())
    input("Press Enter to continue...")

    # Class distributions
    print("Class distributions")
    print(dataset.groupby('class').size())
    input("Press Enter to continue...")

if run_visualisation:
    # Box and whisker plots
    print("Box and Whisker")
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    # Histograms
    print("Histograms")
    dataset.hist()
    plt.show()

    # Scatter plot matrix
    print("Scatter plot matrix")
    scatter_matrix(dataset)
    plt.show()


if run_evaluation:
    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    # Test options and evaluation metric
    seed = 1203812
    n_splits = 10
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    print("Press Enter to continue...")

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

if run_prediction:

    if not run_evaluation:
        # Split-out validation dataset
        array = dataset.values
        X = array[:, 0:4]
        Y = array[:, 4]
        validation_size = 0.20
        seed = 121865
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)

    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

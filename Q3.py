import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
import csv

# Load the Mushroom dataset from UCI ML Repository
def load_data():
    # Code courtesy of Prof. Ioannidis
    filename = './datasets/mushroom/agaricus-lepiota.data'


    # Learn the names of all categories present in the dataset,
    # and map them to 0,1,2,...

    col_maps = {}


    print("Processing",filename,"...",end="")
    with open(filename) as csvfile:
        fr = csv.reader(csvfile, delimiter=',')
        rows = 0
        for row in fr:
            rows += 1
            if rows == 1:
                columns = len(row)
                for c in range(columns):
                    col_maps[c] = {}

            for (c,label) in enumerate(row):
                if label not in col_maps[c]:
                    index = len(col_maps[c])
                    col_maps[c][label] = index
    print(" done")

    print("Read %d rows having %d columns." % (rows,columns))
    print("Category maps:")
    for c in range(columns):
        print("\t Col %d: " % c, col_maps[c])



    # Construct matrix X, containing the mapped
    # features, and vector y, containing the mapped
    # labels.

    X = []
    y = []

    print("Converting",filename,"...",end="")
    with open(filename) as csvfile:
        fr = csv.reader(csvfile, delimiter=',')
        for row in fr:
            label = row[0]
            y.append(col_maps[0][label])

            features = []
            for (c,label) in enumerate(row[1:]):
                features.append(col_maps[c+1][label])

            X.append(features)

    print("done")


    # Store them to files.

    with open('X_msrm.csv', 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        for features in X:
            fw.writerow(features)

    with open('y_msrm.csv', 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        for label in y:
            fw.writerow([label])
    return X, y, col_maps

def optimize_hyperparameters(X_train, X_test, y_train, y_test, col_maps):
    # Run a Categorical Naive Bayes classifier for alphas in the range [2^-15, 2^5]
    # Return scores of test with each value of alpha, and the best alpha and its AUC
    min_categories = [len(col_maps[i]) for i in range(1, len(col_maps))]

    alpha_min = 2**-15
    alpha_max = 2**5

    alpha = alpha_min
    max_auc = 0
    max_alpha_auc = -1
    scores = []
    alphas = []

    while alpha <= alpha_max:
        classifier = CategoricalNB(alpha=alpha, min_categories=min_categories)
        print("Training classifier on alpha = ", alpha)
        classifier.fit(X_train, y_train)
        print("done training, predicting...")

        y_pred = classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: ", acc, "alpha = ", alpha)

        f1 = f1_score(y_test, y_pred)
        print("F1 Score: ", f1, "alpha = ", alpha)

        probs = classifier.predict_proba(X_test)

        roc = roc_auc_score(y_test, probs[:, 1])
        print("AUC: ", roc, "alpha = ", alpha)

        scores.append([acc, roc, f1])

        if(roc > max_auc):
            max_auc = roc
            max_alpha_auc = alpha
        alphas.append(alpha)
        alpha = alpha * 2
        print()

    print("Max AUC: ", max_auc, " at alpha = ", max_alpha_auc)
    return scores, alphas, max_auc, max_alpha_auc


if __name__ == '__main__':
    # Seed (42 is a wise choice on a cosmic scale)
    random_state = 42
    X, y, col_maps = load_data()

    classes = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Get results for 20-80 split
    scores, alphas, max_auc, max_alpha_auc = optimize_hyperparameters(X_train, X_test, y_train, y_test, col_maps)

    legend = ['Accuracy', 'AUC', 'F1 Score']
    plt.plot(alphas, scores, label=legend)
    plt.legend()
    plt.title("20-80 test-train Split")
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.gca().set_xscale('log')
    plt.savefig("8020_split.png")

    min_categories = [len(col_maps[i]) for i in range(1, len(col_maps))]
    classifier = CategoricalNB(alpha=max_alpha_auc, min_categories=min_categories)
    classifier.fit(X_train, y_train)
    print("Classifier Parameters: 80-20 split", classifier.feature_log_prob_)

    parameters = classifier.feature_log_prob_
    for i in range(len(classes)):
        print(classes[i])
        print('poisonous', parameters[i][0])
        print('edible', parameters[i][1])
        print()
        import csv

        # Write classifier parameters to a CSV file
        with open('80-20-params.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class', 'Poisonous', 'Edible'])
            for i in range(len(classes)):
                writer.writerow([classes[i], parameters[i][0], parameters[i][1]])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=random_state)

    # Get results for 99-01 split
    scores, alphas, max_auc, max_alpha_auc = optimize_hyperparameters(X_train, X_test, y_train, y_test, col_maps=col_maps)
    plt.cla()
    plt.plot(alphas, scores, label=legend)
    plt.title("99-1 test-train Split")
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.legend()
    plt.gca().set_xscale('log')
    plt.savefig("0199_split.png")

    min_categories = [len(col_maps[i]) for i in range(1, len(col_maps))]
    classifier = CategoricalNB(alpha=max_alpha_auc, min_categories=min_categories)
    classifier.fit(X_train, y_train)
    print("Classifier Parameters: 01-99 split")
    print(len(classifier.feature_log_prob_))
    print(np.shape(classifier.feature_log_prob_[1]))


    parameters = classifier.feature_log_prob_
    for i in range(len(classes)):
        print(classes[i])
        print('poisonous', parameters[i][0])
        print('edible', parameters[i][1])
        import csv

        # Write classifier parameters to a CSV file
        with open('01-99-params.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class', 'Poisonous', 'Edible'])
            for i in range(len(classes)):
                writer.writerow([classes[i], parameters[i][0], parameters[i][1]])

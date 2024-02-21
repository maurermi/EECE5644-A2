import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

def load_data():
    # Courtesy Prof. Ioannidis
    import csv
    import os

    def to_lower_case(s):
        """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'.
        """
        return s.lower()

    def strip_non_alpha(s):
        """ Remove non-alphabetic characters from the beginning and end of a string.

        E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle
        of the string should not be removed. E.g. "haven't" should remain unaltered."""

        s = s.strip()
        if len(s)==0:
            return s
        if not s[0].isalpha():
            return strip_non_alpha(s[1:])
        elif not s[-1].isalpha():
            return strip_non_alpha(s[:-1])
        else:
            return s

    def clean(s):
        """ Create a "clean" version of a string
        """
        return to_lower_case(strip_non_alpha(s))


    # Directory of text files to be processed

    directory = './datasets/sentence+classification/SentenceCorpus/labeled_articles/'



    # Learn the vocabulary of words in the corpus
    # as well as the categories of labels used per text

    categories = {}
    vocabulary = {}


    num_files = 0
    for filename in [x for x in os.listdir(directory) if ".txt" in x]:
        num_files +=1
        print("Processing",filename,"...",end="")
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f,'r') as  fp:
                for line in fp:
                    line = line.strip()
                    if "###" in line:
                        continue
                    if "--" in line:
                        label, words = line.split("--")
                        words = [clean(word) for word in words.split()]
                    else:
                        words = line.split()
                        label = words[0]
                        words = [clean(word) for word in words[1:]]

                    if label not in categories:
                        index = len(categories)
                        categories[label] = index

                    for word in words:
                        if word not in vocabulary:
                            index = len(vocabulary)
                            vocabulary[word] = index
        print(" done")

    n_words = len(vocabulary)
    n_cats = len(categories)

    print("Read %d files containing %d words and %d categories" % (num_files,len(vocabulary),len(categories)))

    print(categories)


    # Convert sentences into a "bag of words" representation.
    # For example, "to be or not to be" is represented as
    # a vector with length equal to the vocabulary size,
    # with the value 2 at the indices corresponding to "to" and "be",
    # value 1 at the indices corresponding to "or" and "not"
    # and zero everywhere else.


    X = []
    y = []

    for filename in [x for x in os.listdir(directory) if ".txt" in x]:
        print("Converting",filename,"...",end="")
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f,'r') as  fp:
                for line in fp:
                    line = line.strip()
                    if "###" in line:
                        continue
                    if "--" in line:
                        label, words = line.split("--")
                        words = [clean(word) for word in words.split()]
                    else:
                        words = line.split()
                        label = words[0]
                        words = [clean(word) for word in words[1:]]

                    y.append(categories[label])

                    features = n_words * [0]

                    bag = {}
                    for word in words:
                        if word not in bag:
                            bag[word] = 1
                        else:
                            bag[word] += 1

                    for word in bag:
                        features[vocabulary[word]] = bag[word]

                    X.append(features)
        print(" done")

    # Save X and y to files

    with open('X_snts.csv', 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        for features in X:
            fw.writerow(features)

    with open('y_snts.csv', 'w') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        for label in y:
            fw.writerow([label])

    return X, y, categories, vocabulary, n_words, n_cats


def optimize_hyperparameters(X_train, X_test, y_train, y_test, alphas):
    # Run classifier for values of alpha in alphas
    # Return the accuracy score for each value of alpha,
    # the best accuracy, the best alpha, and the feature log probabilities for the best alpha
    scores = []
    max_accuracy = 0
    max_alpha_accuracy = -1

    log_probs_best = []

    for alpha in alphas:
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        if score > max_accuracy:
            max_accuracy = score
            max_alpha_accuracy = alpha
            log_probs_best = classifier.feature_log_prob_
        scores.append(score)
    print(classifier.classes_)
    print("Best accuracy", max_accuracy, "Best alpha", max_alpha_accuracy)
    return scores, max_accuracy, max_alpha_accuracy, log_probs_best


def run_test(X, y):
    # Run 10 tests with different random splits of the data
    # Plot average accuracy for each value of alpha among the 10 tests
    # Return the log probabilities for the best performing value of alpha
    alphas = [2**i for i in range(-15, 6)]
    print(alphas)

    scores = np.zeros(shape=(10, 21))

    log_probs_best_dict = dict()

    for seed in range(10):
        print("Seed: ", seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        res = optimize_hyperparameters(X_train, X_test, y_train, y_test, alphas)
        scores[seed] = res[0]
        log_probs_best_dict[res[2]] = res[3]


    avg_score_for_alpha = np.mean(scores, axis=0)
    stdev_for_alpha = np.std(scores, axis=0)

    import matplotlib.pyplot as plt
    plt.errorbar(alphas, avg_score_for_alpha, yerr=stdev_for_alpha)
    plt.gca().set_xscale('log')
    plt.title("Average Accuracy vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Average Accuracy")
    plt.savefig("Q4.png")
    # plt.show()

    best_average_alpha = np.argmax(avg_score_for_alpha)
    print("Best Average Alpha: ", alphas[best_average_alpha], "Average:", avg_score_for_alpha[best_average_alpha])
    return log_probs_best_dict, alphas, best_average_alpha

if __name__ == '__main__':
    X, y, categories, vocabulary, n_words, n_cats = load_data()

    res = run_test(X, y)

    labels = ['MISC', 'AIMX', 'OWNX', 'CONT', 'BASE']
    log_probs = res[0][res[1][res[2]]]
    # Print the five most probable words for each class
    # (i.e. the words with the largest posterior)
    for idx in range(5):
        print(labels[idx], np.argpartition(log_probs[idx], -5)[-5:])
    print()
    # Print the ten most probable words for each class
    # (i.e. the words with largest posterior)
    for idx in range(5):
        print(labels[idx], np.argpartition(log_probs[idx], -10)[-10:])
    print()
    print(categories)
    print()
    print(vocabulary)

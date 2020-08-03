import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split(' ')
    norm_words = []
    for idx,wrd in enumerate(words):
        norm_words.append(wrd.lower()) if wrd != '' else None
    return norm_words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    #add all words to dictionary with a count for each of how many messages they appear in
    word_count_dict = {}
    for message in messages:
        words = get_words(message)
        words_appeared = set() #use set to store words seen in message so far
        for word in words:
            if word in word_count_dict and word not in words_appeared:
                words_appeared.add(word)
                word_count_dict[word] = word_count_dict[word] + 1
            if word not in word_count_dict:
                words_appeared.add(word)
                word_count_dict[word] = 1


    #add words from previous dictionary if appear in 5+ messages
    word_dict = {}
    i=0
    for word,ct in word_count_dict.items():
        if ct>=5:
            word_dict[word] = i #add to spam_dictionary
            i+=1
    return word_dict


    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    vocab_size = len(word_dictionary)
    num_messages = len(messages)

    #initialize array to 0
    arr = np.zeros((num_messages,vocab_size))

    for idx,message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                arr[idx][word_dictionary[word]]+=1

    return arr
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    #compute phi_y estimate
    phi_y = np.mean(np.array(labels))
    #compute posterior probabilities
    phi_given_pos = np.ones(matrix.shape[1]) #since Laplace smoothing
    phi_given_neg = np.ones(matrix.shape[1])

    pos_denom = matrix.shape[1] #since Laplace smoothing
    neg_denom = matrix.shape[1]

    for i in range(matrix.shape[0]):
        pos = (labels[i]==1)
        if pos:
            pos_denom+= np.sum(matrix[i]) #add d
        else:
            neg_denom+= np.sum(matrix[i]) #add d


        #add number of that vocab that appear
        if pos:
            phi_given_pos += matrix[i]
        else:
            phi_given_neg += matrix[i]

    phi_given_pos = phi_given_pos/pos_denom
    phi_given_neg = phi_given_neg/neg_denom

    #store all 3 in a dictionary
    dict_ = {}
    dict_['phi_y'] = phi_y
    dict_['phi_pos'] = phi_given_pos
    dict_['phi_neg'] = phi_given_neg

    return dict_

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    # Note that the model is a dictionary with phi_y, phi_pos and phi_neg
    phi_y = model['phi_y']
    phi_pos = model['phi_pos']
    phi_neg = model['phi_neg']

    preds = []
    for x in range(matrix.shape[0]): #go through each test data
        p_1 = np.log(phi_y)
        p_0 = np.log(1-phi_y)
        for v in range(matrix.shape[1]):
            for i in range(int(matrix[x][v])):
                p_1+= np.log(phi_pos[v]) #add all log probabilities
                p_0+= np.log(phi_neg[v])
        preds.append(1) if p_1 > p_0 else preds.append(0)

    return np.array(preds)
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    # Note that the model is a dictionary with phi_y, phi_pos and phi_neg
    phi_pos = model['phi_pos']
    phi_neg = model['phi_neg']

    metric = np.log(phi_pos) - np.log(phi_neg)
    top_five_args = np.flip(np.argsort(metric))[:5]
    top_five = []
    for arg in top_five_args:
        for key,val in dictionary.items():
            if(val==arg):
                top_five.append(key)


    return top_five

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    top_accuracy = 0
    top_radi = 0
    for radi in radius_to_consider:
        cur_preds = svm.train_and_predict_svm(train_matrix,train_labels,val_matrix,radi)
        acc = np.mean(cur_preds==val_labels)
        if acc >= top_accuracy:
            top_radi = radi

    return top_radi

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()

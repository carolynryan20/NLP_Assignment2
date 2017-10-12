import nltk
from decimal import Decimal
import numpy as np 
import matplotlib.pyplot as plt

def main():
    '''
    Open file, clean it, make and evaluate models, run sentence probability
    :return: None
    '''
    f = open("HW2_InputFile.txt", "r")
    line = f.readline()
    corpus = clean_corpus(line)
    sentence_tokens = nltk.tokenize.sent_tokenize(corpus)
    word_tokens = tokenize_words(corpus)
    unique_word_dict = unique_words(word_tokens)

    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", word_count(word_tokens))

    plot_word_freq(unique_word_dict)

    ngram_model_and_analysis(sentence_tokens)

    print()
    sentence_mle(sentence_tokens)
    plt.show()
    f.close()


def ngram_model_and_analysis(sentence_tokens):
    '''
    Make unigram, bigram, trigram models with different cuts
    :param sentence_tokens: A tokenized list of sentences
    :return: None
    '''

    # Want beginning and ending symbols for our bigram and trigram models
    bt_sent_tokens = []
    for i in range(len(sentence_tokens)):
        bt_sent_tokens.append( "BEGINNING " + sentence_tokens[i][:-1] + " ENDING")

    print()
    u_train_on_split(.7, sentence_tokens)
    u_train_on_split(.8, sentence_tokens)
    u_train_on_split(.9, sentence_tokens)

    print()
    b_train_on_split(.7, bt_sent_tokens)
    b_train_on_split(.8, bt_sent_tokens)
    b_train_on_split(.9, bt_sent_tokens)

    print()
    t_train_on_split(.7, bt_sent_tokens)
    t_train_on_split(.8, bt_sent_tokens)
    t_train_on_split(.9, bt_sent_tokens)


def plot_word_freq(unique_word_dict):
    '''
    Plot word frequencies, simple visualization with no word labels
    :param unique_word_dict: Dictionary and count of all words in the corpus
    :return: A plot of the word frequencies
    '''
    objects = unique_word_dict.keys()
    y_pos = np.arange(len(objects))
    values = sorted(unique_word_dict.values())[::-1]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks([])
    plt.ylabel('Word Frequency')
    plt.title('Word Frequency in Twin Peaks Review')
    return plt

def clean_corpus(corpus):
    '''
    Documentation and replacement for adjustments we made to the corpus
    :param corpus: A string of the corpus
    :return: A cleaned version of the corpus, string
    '''
    corpus = corpus.replace(".I ", ". I ")
    corpus = corpus.replace(".It ", ". It ")
    corpus = corpus.replace(".That ", ". That ")
    corpus = corpus.replace("Mr. ", "Mr ")
    corpus = corpus.replace("Dr. ", "Dr ")
    corpus = corpus.replace("F.B.I.", "FBI")
    corpus = corpus.replace("Catherine E.", "Catherine E")
    corpus = corpus.replace("ä", "a")

    corpus = corpus.replace("'re", " are")
    corpus = corpus.replace("'ve", " have")
    corpus = corpus.replace("n't", " not")
    corpus = corpus.replace("I'm", "I am")
    corpus = corpus.replace("It's", "It is")
    corpus = corpus.replace("it's", "it is")
    corpus = corpus.replace("What's", "What is")
    corpus = corpus.replace("what's", "what is")
    corpus = corpus.replace("There's", "There is")
    corpus = corpus.replace("there's", "there is")
    corpus = corpus.replace("that's", "that is")
    corpus = corpus.replace("That's", "That is")
    corpus = corpus.replace("He's", "He is") # sometimes awkward tense, as should be 'he has' in some cases
    corpus = corpus.replace("he's", "he is")
    corpus = corpus.replace("Let's", "Let us")
    corpus = corpus.replace("\"", "")

    return corpus

def tokenize_words(corpus):
    '''
    Returns a list of all words and punctuations in the corpus
    :param corpus: A string of the corpus
    :return: A list of all words
    '''
    word_tokens = nltk.word_tokenize(corpus)
    new_word_tokens = []
    for word_index in range(len(word_tokens)):
        word = word_tokens[word_index]
        if word == "'s":
            new_word_tokens[-1] += word
        else:
            new_word_tokens.append(word)
    return new_word_tokens

def word_count(word_tokens):
    '''
    Returns word count
    :param word_tokens: A list of words
    :return: int count of all words (non-inclusive to punctuation)
    '''
    count = 0
    for word in word_tokens:
        if len(word) > 1 or word.isalpha(): # Discount punctuation
            count += 1

    return count

def unique_words(word_tokens):
    '''
    Identify and count occurence of all words
    :param word_tokens: A list of words
    :return:  A dictionary of format word: count
    '''
    unique = dict()
    for word in word_tokens:
        word = word.lower()
        if unique.get(word):
            unique[word] = unique[word] + 1.0
        else:
            unique[word] = 1.0
    return unique

def split_data(cut, tokens):
    '''
    Splits data into training and test given a cut
    http://www.nltk.org/book/ch04.html
    :param cut: Float of desired training percentage
    :param tokens: List to cut
    :return: Training and test data
    '''
    cut = int(cut*len(tokens))
    training_data, test_data = tokens[:cut], tokens[cut:]
    return training_data, test_data

def u_train_on_split(cut, tokens, test_sentence=None):
    '''
    Unigram Model: Training and Evaluation (Perplexity or MLE)
    :param cut: float of desired cut
    :param tokens: List of words
    :param test_sentence: (Optional) instead of test data use only sentence, get MLE not perplexity
    :return: None
    '''
    training_data, test_data = split_data(cut, tokens)
    if test_sentence:
        test_data = [test_sentence]

    training_counts = unique_words(' '.join(training_data))
    training_total = len(' '.join(training_data))
    for key, value in training_counts.items():
        training_counts[key] = value/training_total

    sentence_perplexity = []
    sentence_probability = []
    for sentence in test_data:
        prob = Decimal(1.0)
        word_tokens = tokenize_words(sentence)
        for word in word_tokens:
            if training_counts.get(word):
                prob = prob * Decimal(training_counts[word])
            else:
                prob = prob * Decimal(.000001)
        sentence_perplexity.append(prob**(Decimal(-1/len(sentence))))
        sentence_probability.append(prob)

    if test_sentence:
        print("Unigram MLE is", sum(sentence_probability)/len(test_data))
    else:
        perplexity = sum(sentence_perplexity) / len(test_data)
        print("Unigram perplexity ({}/{} split): {}".format(
         round(cut*100), round((1-cut)*100), round(perplexity, 5)))

def b_train_on_split(cut, tokens, test_sentence=None):
    '''
    Bigram Model: Training and Evaluation (Perplexity or MLE)
    :param cut: float of desired cut
    :param tokens: List of words
    :param test_sentence: (Optional) instead of test data use only sentence, get MLE not perplexity
    :return: None
    '''
    training_data, test_data = split_data(cut, tokens)
    if test_sentence:
        test_data = [test_sentence]

    training_word_tokens = tokenize_words(' '.join(training_data))
    training_total = len(training_word_tokens)

    bigram_count = dict()
    for i in range(len(training_word_tokens)-1):
        key = training_word_tokens[i].lower()+" "+training_word_tokens[i+1].lower()
        if key in bigram_count:
            bigram_count[key] += 1
        else:
            bigram_count[key] = 1

    for key,value in bigram_count.items():
        bigram_count[key] = value/training_total

    sentence_perplexity = []
    sentence_probability = []
    for sentence in test_data:
        prob = Decimal(1.0)
        word_tokens = tokenize_words(sentence)
        for word_index in range(len(word_tokens) -1):
            bigram = word_tokens[word_index] + " " + word_tokens[word_index + 1]
            if bigram_count.get(bigram):
                prob = prob * Decimal(bigram_count[bigram])
            else:
                prob = prob * Decimal(.000001)
        sentence_perplexity.append(prob**(Decimal(-1/len(sentence))))
        sentence_probability.append(prob)

    if test_sentence:
        print("Bigram MLE is", sum(sentence_probability) / len(test_data))
    else:
        perplexity = sum(sentence_perplexity) / len(test_data)
        print("Bigram perplexity ({}/{} split): {}".format(
            round(cut * 100), round((1 - cut) * 100), round(perplexity, 5)))

def t_train_on_split(cut, tokens, test_sentence=None):
    '''
    Trigram Model: Training and Evaluation (Perplexity or MLE)
    :param cut: float of desired cut
    :param tokens: List of words
    :param test_sentence: (Optional) instead of test data use only sentence, get MLE not perplexity
    :return: None
    '''
    training_data, test_data = split_data(cut, tokens)
    if test_sentence:
        test_data = [test_sentence]

    training_word_tokens = tokenize_words(' '.join(training_data))
    training_total = len(training_word_tokens)

    trigram_count = dict()
    for i in range(len(training_word_tokens) - 2):
        key = training_word_tokens[i].lower() + " " + training_word_tokens[i + 1].lower() + " " + training_word_tokens[i+2]
        if key in trigram_count:
            trigram_count[key] += 1
        else:
            trigram_count[key] = 1

    for key,value in trigram_count.items():
        trigram_count[key] = value/training_total

    sentence_perplexity = []
    sentence_probability = []
    for sentence in test_data:
        prob = Decimal(1.0)
        word_tokens = tokenize_words(sentence)
        for word_index in range(len(word_tokens) - 2):
            trigram = word_tokens[word_index] + " " + word_tokens[word_index + 1] + " " + training_word_tokens[word_index+2]
            if trigram_count.get(trigram):
                prob = prob * Decimal(trigram_count[trigram])
            else:
                prob = prob * Decimal(.000001)
        sentence_probability.append(prob)
        sentence_perplexity.append(prob ** (Decimal(-1 / len(sentence))))

    if test_sentence:
        print("Trigram MLE is", sum(sentence_probability) / len(test_data))
    else:
        perplexity = sum(sentence_perplexity) / len(test_data)
        print("Trigram perplexity ({}/{} split): {}".format(
            round(cut * 100), round((1 - cut) * 100), round(perplexity, 5)))

def sentence_mle(sentence_tokens):
    '''
    Using input file of sentences.txt, calculate MLE of our best ngram models on those sentences
    :param sentence_tokens: A list of all sentences in the corpus
    :return: None
    '''
    f = open("sentences.txt", "r")
    bt_sent_tokens = []
    for i in range(len(sentence_tokens)):
        bt_sent_tokens.append("BEGINNING "+sentence_tokens[i][:-1] + " ENDING")

    for line in f.readlines():
        print()
        print(line[:-1])
        u_train_on_split(.9, sentence_tokens, line)
        b_train_on_split(.9, bt_sent_tokens, line)
        t_train_on_split(.9, bt_sent_tokens, line)

    f.close()

if __name__ == '__main__':
    main()


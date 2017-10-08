import nltk

# Changes to HW2_InputFile:
    # The textfile used non-ascii apostrophes and quotations
    #   so we changed these to their ASCII counterparts


def main():
    f = open("HW2_InputFile.txt", "r")
    line = f.readline()
    corpus = clean_corpus(line)
    sentence_tokens = nltk.tokenize.sent_tokenize(corpus) # Doesn't work on 'F.B.I agent Dale' splits into two sentences
    word_tokens = tokenize_words(corpus)
    unique_word_dict = unique_words(word_tokens)

    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", word_count(word_tokens))
    u_train_on_split(.7, word_tokens)
    u_train_on_split(.8, word_tokens)
    u_train_on_split(.9, word_tokens)
    f.close()

def clean_corpus(corpus):
    # Documentation on what we changed to the corpus
    corpus = corpus.replace(".I ", ". I ")
    corpus = corpus.replace(".It ", ". It ")
    corpus = corpus.replace(".That ", ". That ")
    corpus = corpus.replace("Mr. ", "Mr ")
    corpus = corpus.replace("Dr. ", "Dr ")
    corpus = corpus.replace("F.B.I", "FBI")
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

    return corpus

def tokenize_words(corpus):
    word_tokens = nltk.word_tokenize(corpus)
    new_word_tokens = []
    wordAlreadyAdded = False
    for word_index in range(len(word_tokens)):
        word = word_tokens[word_index]
        if word == "'":
            new_word_tokens[-1] += word + word_tokens[word_index+1]
            wordAlreadyAdded = True
        elif wordAlreadyAdded:
            wordAlreadyAdded = False
        else:
            new_word_tokens.append(word)
            wordAlreadyAdded = False
    return new_word_tokens

def word_count(word_tokens):
    count = 0
    for word in word_tokens:
        if len(word) > 1 or word.isalpha(): # Discount punctuation
            count += 1

    return count

def unique_words(word_tokens):
    unique = dict()
    for word in word_tokens:
        word = word.lower()
        if unique.get(word):
            unique[word] = unique[word] + 1.0
        else:
            unique[word] = 1.0
    return unique

def split_data(cut, tokens):
    # http://www.nltk.org/book/ch04.html
    cut = int(cut*len(tokens))
    training_data, test_data = tokens[:cut], tokens[cut:]
    return training_data, test_data

def u_train_on_split(cut, tokens):
    training_data, test_data = split_data(cut, tokens)

    # UNIGRAM
    training_counts = unique_words(' '.join(training_data))
    training_total = len(training_data)
    for key, value in training_counts.items():
        training_counts[key] = value/training_total

    sentence_probs = []
    for sentence in training_data:
        prob = 1.0
        for word in sentence:
            if training_counts.get(word):
                prob = prob*training_counts[word]
            else:
                prob = prob*.000001
        sentence_probs.append(prob)

    perplexity = sum(sentence_probs)**(-1/training_total)
    print ("Unigram perplexity: " + str(perplexity))

if __name__ == '__main__':
    main()

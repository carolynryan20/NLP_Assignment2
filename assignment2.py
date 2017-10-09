import nltk
from decimal import Decimal

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
    # print(word_tokens)

    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", word_count(word_tokens))
    u_train_on_split(.7, sentence_tokens)
    u_train_on_split(.8, sentence_tokens)
    u_train_on_split(.9, sentence_tokens)
    print()
    b_train_on_split(.7, sentence_tokens)
    b_train_on_split(.8, sentence_tokens)
    best_bigram = b_train_on_split(.9, sentence_tokens)

    print()
    t_train_on_split(.7, sentence_tokens)
    t_train_on_split(.8, sentence_tokens)
    t_train_on_split(.9, sentence_tokens)

    print()
    sentence_probabilities(best_bigram)

    f.close()

def clean_corpus(corpus):
    # Documentation on what we changed to the corpus
    # TODO see why `` is appearing in the corpus????
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

    # TODO add in markers for <s> and </s>
    # corpus = "BEGINNING " + corpus
    # corpus = corpus.replace(".", " ENDING BEGINNING")
    # corpus = corpus.replace("!", " ENDING BEGINNING")
    # corpus = corpus.replace("?", " ENDING BEGINNING")
    return corpus

def tokenize_words(corpus):
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
    training_total = len(' '.join(training_data))
    for key, value in training_counts.items():
        training_counts[key] = value/training_total

    sentence_probs = []
    for sentence in test_data:
        prob = Decimal(1.0)
        for word in sentence:
            if training_counts.get(word):
                prob = prob * Decimal(training_counts[word])
            else:
                prob = prob * Decimal(.000001)
        sentence_probs.append(prob**(Decimal(-1/len(sentence))))

    perplexity = sum(sentence_probs)/len(test_data)
    print("Unigram perplexity ({}/{} split): {}".format(
         round(cut*100), round((1-cut)*100), round(perplexity, 5)))

    return training_counts

def b_train_on_split(cut, tokens):
    training_data, test_data = split_data(cut, tokens)
    training_word_tokens = tokenize_words(' '.join(training_data))
    training_total = len(training_word_tokens)

    bigram_count = dict()
    for i in range(len(training_word_tokens)-1):
        key = training_word_tokens[i].lower()+" "+training_word_tokens[i+1].lower()
        if key in bigram_count:
            bigram_count[key] += 1
        else:
            bigram_count[key] = 1
    # print(bigram_count)

    for key,value in bigram_count.items():
        bigram_count[key] = value/training_total

    sentence_perplexity = []
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

    perplexity = sum(sentence_perplexity) / len(test_data)
    print("Bigram perplexity ({}/{} split): {}".format(
        round(cut * 100), round((1 - cut) * 100), round(perplexity, 5)))

    return bigram_count

def t_train_on_split(cut, tokens):
    training_data, test_data = split_data(cut, tokens)
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
    for sentence in test_data:
        prob = Decimal(1.0)
        word_tokens = tokenize_words(sentence)
        for word_index in range(len(word_tokens) - 2):
            trigram = word_tokens[word_index] + " " + word_tokens[word_index + 1] + " " + training_word_tokens[word_index+2]
            if trigram_count.get(trigram):
                prob = prob * Decimal(trigram_count[trigram])
            else:
                prob = prob * Decimal(.000001)
        sentence_perplexity.append(prob ** (Decimal(-1 / len(sentence))))

    perplexity = sum(sentence_perplexity) / len(test_data)
    print("Trigram perplexity ({}/{} split): {}".format(
        round(cut * 100), round((1 - cut) * 100), round(perplexity, 5)))
    return trigram_count

def sentence_probabilities(best_bigram):
    f = open("sentences.txt", "r")
    all_lines = f.readlines()
    clean_lines = clean_corpus(' '.join(all_lines))
    sentence_tokens = nltk.tokenize.sent_tokenize(clean_lines)

    for sentence in sentence_tokens:
        prob = Decimal(1.0)
        word_tokens = tokenize_words(sentence)
        for word_index in range(len(word_tokens) - 1):
            bigram = word_tokens[word_index] + " " + word_tokens[word_index + 1]
            if best_bigram.get(bigram):
                prob = prob * Decimal(best_bigram[bigram])
            else:
                prob = prob * Decimal(.000001)
        perplexity = prob ** (Decimal(-1 / len(sentence)))
        print("Perplexity is {} for {}".format(round(perplexity, 5), sentence))
    f.close()


if __name__ == '__main__':
    main()

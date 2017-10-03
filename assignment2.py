import nltk
# nltk.download('punkt')

# Changes to HW2_InputFile:
    # Added space between .It for 'It was a hell of a payoff'
    # Addedd space between '.I love that he and'
    # Separate all contractions
    # Get rid of non sentence ending '.'


def main():
    f = open("HW2_InputFile.txt", "r")
    line = f.readline()
    corpus = clean_corpus(line)
    sentence_tokens = nltk.tokenize.sent_tokenize(corpus) # Doesn't work on 'F.B.I agent Dale' splits into two sentences
    word_tokens = tokenize_words(corpus)

    print("Entire text (cleaned):", corpus)
    print("NLTK Sentence Tokenizer:", sentence_tokens)
    print("NLTK Word Tokenizer:", word_tokens,"\n")


    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", word_count(word_tokens))
    f.close()

def clean_corpus(corpus):
    # Documentation on what we changed to the corpus
    corpus = corpus.replace(".I ", ". I ")
    corpus = corpus.replace(".It ", ". It ")
    corpus = corpus.replace("Mr. ", "Mr ")
    corpus = corpus.replace("Dr. ", "Dr ")
    corpus = corpus.replace("F.B.I", "FBI")
    corpus = corpus.replace("Catherine E.", "Catherine E")

    corpus = corpus.replace("’re", " are")
    corpus = corpus.replace("’ve", " have")
    corpus = corpus.replace("n’t", " not")
    corpus = corpus.replace("I’m", "I am")
    corpus = corpus.replace("It’s", "It is")
    corpus = corpus.replace("it’s", "it is")
    corpus = corpus.replace("What’s", "What is")
    corpus = corpus.replace("what’s", "what is")
    corpus = corpus.replace("There’s", "There is")
    corpus = corpus.replace("there’s", "there is")
    corpus = corpus.replace("that’s", "that is")
    corpus = corpus.replace("That’s", "That is")
    corpus = corpus.replace("He’s", "He is")
    corpus = corpus.replace("he’s", "he is")
    corpus = corpus.replace("Let’s", "Let us")

    return corpus

def tokenize_words(corpus):
    word_tokens = nltk.word_tokenize(corpus)
    new_word_tokens = []
    wordAlreadyAdded = False
    for word_index in range(len(word_tokens)):
        word = word_tokens[word_index]
        if word == "’":
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


if __name__ == '__main__':
    main()


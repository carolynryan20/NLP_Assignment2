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
    sentence_tokens = nltk.tokenize.sent_tokenize(line) # Doesn't work on 'F.B.I agent Dale' splits into two sentences
    word_tokens = nltk.word_tokenize(line)

    print("Entire text:", line)
    print("NLTK Sentence Tokenizer:", sentence_tokens)
    print("NLTK Word Tokenizer:", word_tokens,"\n")


    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", len(word_tokens))
    f.close()

if __name__ == '__main__':
    main()


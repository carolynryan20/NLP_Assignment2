import nltk
# nltk.download('punkt')

# Changes to HW2_InputFile:
    # Added space between .It for 'It was a hell of a payoff'

def main():
    f = open("HW2_InputFile.txt", "r")
    line = f.readline()
    sentence_tokens = nltk.tokenize.sent_tokenize(line) # Doesn't work on 'F.B.I agent Dale' splits into two sentences
    print("Entire text:", line)
    print("NLTK Sentence Tokenizer:", sentence_tokens)

    new_line = ""
    new_sentence_tokens = []
    for sentence in sentence_tokens:
        if sentence[0] in "abcdefghijklmnopqrstuvwxyz":
            new_line = new_line[0:-6] + ". "
            new_line += sentence + " </s> "
            new_sentence_tokens[-1] += " "+sentence
        else:
            new_line += "<s> " + sentence[0:-1] + " </s> "
            new_sentence_tokens.append(sentence)
    print("Entire text with start stop:", new_line)
    print("Entire text with new sentence measures:", new_sentence_tokens)

    tokens = nltk.word_tokenize(new_line)
    print("NLTK Word Tokenizer:", tokens)

    words = new_line.split(" ")
    print("Words split at ' ':", words)

    print("The number of sentences in the text is", len(sentence_tokens))
    print("The number of words in the text is", len(words))
    f.close()



if __name__ == '__main__':
    main()
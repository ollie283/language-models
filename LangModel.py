import re

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_sentences_from_file():
    with open("./sampledata.txt", "r") as f:
        print ([re.split("\s+", line.rstrip('\n')) for line in f])

read_sentences_from_file()

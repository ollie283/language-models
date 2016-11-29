import re

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_sentences_from_file():
    with open("./sampledata.txt", "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]


    
class UnigramLanguageModel:
    def __init__(self, sentences, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing
      
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        # remove two for <s> and </s>
        unigram_count += len(sentence) - 2
    print (unigram_count)

calculate_number_of_unigrams(read_sentences_from_file())

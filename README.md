# Language Models and Smoothing

There are two datasets.

**Toy dataset:** The ﬁles ```sampledata.txt, sampledata.vocab.txt, sampletest.txt``` comprise a small toy dataset. sampledata.txt is the training corpus and contains the following:
```
<s> a a b b c c </s> <s> a c b c </s> <s> b c c a b </s>
```
Treat each line as a sentence. ```<s>``` is the start of sentence symbol and ```</s>``` is the end of sentence symbol. To keep the toy dataset simple, characters a-z will each be considered as a word. i.e. The ﬁrst sentence has 8 tokens, second has 6 tokens, and the last has 7.

The ﬁle ```sampledata.vocab.txt``` contains the vocabulary of the training data. It lists the 3 word types for the toy dataset:
```
a 
b 
c
```
```sampletest.txt``` is the test corpus.

**Actual data:** The ﬁles ```train.txt, train.vocab.txt, and test.txt``` form a larger more realistic dataset. These ﬁles have been pre-processed to remove punctuation and all words have been converted to lower case. An example sentence in the train or test ﬁle has the following form:

```<s> the anglo-saxons called april oster-monath or eostur-monath </s>```

Again every space-separated token is a word. The above sentence has 9 tokens. The ```train.vocab.txt``` contains the vocabulary (types) in the training data.

**Important:** Note that the ```<s>``` or ```</s>``` are not included in the vocabulary ﬁles. The term UNK will be used to indicate words which have not appeared in the training data. UNK is also not included in the vocabulary ﬁles but you will need to add UNK to the vocabulary while doing computations. While computing the probability of a test sentence, any words not seen in the training data should be treated as a UNK token.

**Important:** You do not need to do any further preprocessing of the data. Simply split by space you will have the tokens in each sentence.

## Implementation of the models
a) Write a function to compute unigram unsmoothed and smoothed models. Print out the unigram probabilities computed by each model for the **Toy dataset**.

b) Write a function to compute bigram unsmoothed and smoothed models. Print out the bigram probabilities computed by each model for the **Toy dataset**.

c) Write a function to compute sentence probabilities under a language model. Print out the probabilities of sentences in **Toy dataset** using the smoothed unigram and bigram models.

d) Write a function to return the perplexity of a test corpus given a particular language model. Print out the perplexities computed for ```sampletest.txt``` using a smoothed unigram model and a smoothed bigram model.

## Run on large corpus 
Now use the **Actual dataset**. Train smoothed unigram and bigram models on ```train.txt```. Print out the perplexity under each model for

a) ```train.txt``` i.e. the same corpus you used to train the model.

b) ```test.txt```

## Code
Code should run without any arguments. It should read ﬁles in the same directory. Absolute paths must not be used. 
It should print values in the following format:
```
---------------- Toy dataset ---------------

=== UNIGRAM MODEL ===
 - Unsmoothed  -
a:0.0   b:0.0 ...
- Smoothed -
a:0.0   b:0.0 ...

=== BIGRAM MODEL === 
- Unsmoothed 
a 	b	 c    UNK 	</s> 
a 	0.0  ... 
b	... 
c 	... 
UNK	... 
<s>	...

- Smoothed -
a 	b	 c    UNK 	</s> 
a 	0.0  ... 
b	... 
c 	... 
UNK	... 
<s>	...

== SENTENCE PROBABILITIES == 
sent 		            uprob   biprob 
<s> a b c </s> 	        0.0 	0.0
 <s> a b b c c </s>     ...     ...
 
== TEST PERPLEXITY == 
unigram: 0.0 
bigram: 0.0

---------------- Actual dataset ----------------
PERPLEXITY of train.txt 
unigram: 0.0 
bigram: 0.0

PERPLEXITY of test.txt 
unigram: 0.0 
bigram: 0.0
```

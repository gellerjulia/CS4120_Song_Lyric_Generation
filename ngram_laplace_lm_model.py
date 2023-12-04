from collections import Counter
import numpy as np
import math
import random


# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"

def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  n_grams = []
  for i in range(len(tokens) - n + 1):
    n_grams.append(tuple(tokens[i:i+n]))

  return n_grams

class NGramLaplaceLanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained n-gram language model using laplace smoothing.
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    self.n = n_gram
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # first identify tokens with only one occurrence and replace with UNK 
    token_counts = Counter(tokens)
    cleaned_tokens = []
    for token in tokens:
      if token_counts[token] > 1:
        cleaned_tokens.append(token)
      else:
        cleaned_tokens.append(UNK)

    # find n-grams 
    n_grams = create_ngrams(cleaned_tokens, self.n)
    self.n_gram_counts = Counter(n_grams)

    # get vocabulary for identifying unknown tokens in test data 
    self.vocabulary = set(cleaned_tokens)
    # we'll also want the size of our vocabulary for laplace smoothing calculations 
    self.vocab_size = len(self.vocabulary)

    # collecting data needed for the denominator of score calculations 
    if self.n == 1: 
      # for unigrams, we just need the total number of tokens 
      self.num_tokens = len(cleaned_tokens)
    else: 
      # for other n-grams, we want the n-1 grams for scoring
      self.n_sub1_gram_counts = Counter(create_ngrams(cleaned_tokens, self.n - 1))

    if verbose:
      print("Number of tokens:", len(cleaned_tokens))
      print("N-gram examples:", list(self.n_gram_counts.keys())[:3])
      print("Vocabulary Size:", self.vocab_size)
  
  def score_unigram(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
       Assumes that we are using a unigram model and uses Laplace smoothing.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    total_log_prob = 0
    for token in sentence_tokens:
        unigram_to_score = tuple([token])
        # P(w_i) = ( count(w_i) + 1 ) / ( N + |V| )
        unigram_prob = (self.n_gram_counts[unigram_to_score] + 1) / (self.num_tokens + self.vocab_size)
        total_log_prob += np.log(unigram_prob)

    return np.exp(total_log_prob)

  
  def score_ngram(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
        Assumes that we are using a n-gram model with n > 1 and uses Laplace smoothing.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    n = self.n
    total_log_prob = 0

    # iterate through all n-grams
    for i in range(len(sentence_tokens) - n + 1):
        n_gram_to_score = tuple(sentence_tokens[i:i+n],)
        prefix = tuple(sentence_tokens[i:i+n-1],)

        # P(w_i | w_{i-N+1}...w_{i-1}) = ( count(w_{i-N+1}...w_i) + 1 ) / ( count(w_{i-N+1}...w_{i-1}) + |V| )
        n_gram_prob = (self.n_gram_counts[n_gram_to_score] + 1) / (self.n_sub1_gram_counts[prefix] + self.vocab_size)
        total_log_prob += np.log(n_gram_prob)

    return np.exp(total_log_prob)


  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # replace tokens not in our vocabulary with UNK
    cleaned_sentence_tokens = []
    for token in sentence_tokens:
      if token in self.vocabulary:
        cleaned_sentence_tokens.append(token)
      else:
        cleaned_sentence_tokens.append(UNK)

    if self.n == 1:
      return self.score_unigram(cleaned_sentence_tokens)
    else:
      return self.score_ngram(cleaned_sentence_tokens)

  def generate_sentence_unigram(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
       Assumes that we are using a unigram model.
      
      Returns:
        list: the generated sentence as a list of tokens
    """
    # sentence should start with a SENTENCE_BEGIN
    tokens = [SENTENCE_BEGIN]

    # For the unigram model, just randomly select tokens using their counts in the training data as weights
    # SENTENCE_BEGIN should not be in the middle of our sentence, so exclude it as an option 
    items = [(unigram[0], count) for (unigram, count) in self.n_gram_counts.items() if unigram[0] != SENTENCE_BEGIN]
    choices = [unigram for (unigram, count) in items]
    weights = [count for (token, count) in items]

    # normalize counts into probabilities by dividing by the total count 
    weights = [count / np.sum(weights) for count in weights]
  
    # randomly select tokens until we get a SENTENCE_END
    while tokens[-1] != SENTENCE_END: 
      next_token = np.random.choice(choices, p = weights)
      tokens.append(next_token)

    return tokens
  
  def generate_sentence_ngrams(self) -> list: 
    """Generates a single sentence from a trained language model using the Shannon technique.
       Assumes that we are using a n-gram model with n > 1.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # sentence should start with n - 1 SENTENCE BEGINS
    tokens = ([SENTENCE_BEGIN] * (self.n - 1))

    # we end the sentence when the last n-1 tokens are SENTENCE_END
    while tokens[-1 * (self.n - 1):] != ([SENTENCE_END] * (self.n - 1)): 
      # the prefix to the next generated token is the last n-1 tokens 
      prefix = tokens[-1 * (self.n - 1):]

      # look for all n-grams where the first n-1 tokens match the designated prefix 
      # grab the last token of these n-grams (unless its SENTENCE_BEGIN, which should not appear mid-sentence)
      items = [(ngram[-1], count) for (ngram, count) in self.n_gram_counts.items() if list(ngram[:-1]) == prefix and ngram[-1] != SENTENCE_BEGIN]
      choices = [token for (token, count) in items]
      weights = [count for (token, count) in items]

      # normalize counts into probabilities by dividing by the total count 
      weights = [count / np.sum(weights) for count in weights]

      next_token = np.random.choice(choices, p = weights)
      tokens.append(next_token)

    return tokens

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    if self.n == 1:
      sentence_tokens = self.generate_sentence_unigram()
    else:
      sentence_tokens = self.generate_sentence_ngrams()
      
    # remove <UNK> tokens 
    return list(map(lambda x: x.replace(UNK, ""), sentence_tokens))

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    return [self.generate_sentence() for i in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    score = self.score(sequence)

    # exclude sentence begins in token count for perplexity calculation
    num_tokens = len([token for token in sequence if token != SENTENCE_BEGIN])
    return score ** (-1 / num_tokens)
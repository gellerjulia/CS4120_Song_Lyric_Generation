import nltk
import csv
import pandas as pd

nltk.download('punkt')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"


def clean_artist_name(name: str):
    name = name.lower()
    name = name.replace('-', ' ')
    name = name.replace('/', '')
    return name


def get_lyrics_in_genre(df: pd.DataFrame, genre: str) -> list:
   """
   Returns the lyrics of songs in df with the given genre.

   Args:
      df (pandas DataFrame): dataframe of artist and lyric data
      genre (str): a music genre found in df
    Returns:

   """ 
   genre_df = df[df['genres'].apply(lambda x: genre in x)]
   return genre_df['lyrics'].tolist()


# NOTE: Might not actually need these -- just trying to get things down as I experiment with NLTK's Ngrams 
def tokenize_line(line: str, ngram: int, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  inner_pieces = nltk.word_tokenize(line)

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  return tokens


def generate_ngrams(data: list, ngram: int) -> list:
    '''
    Takes the tokenized lyric data (list of lists) and 
    generates the ngram training samples out of it.
    Parameters:
    data: a list of lists 
    ngram: the size of the ngrams that should be produced 
    return: 
    list of lists in the format [[x1, x2, ... , xn], ...]
    '''
    ngram_samples = []

    for line in data:
        for i in range(len(line) - ngram + 1):
            ngram_samples.append(line[i:i+ngram])
            
    return ngram_samples


# def read_file(datapath, ngram, by_character=False):
#     '''Reads and Returns the "data" as list of list (as shown above)'''
#     data = []
#     with open(datapath, encoding="utf8") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             # THIS IS WHERE WE GET CHARACTERS INSTEAD OF WORDS
#             # replace spaces with underscores
#             data.append(tokenize_line(row['text'].lower(), ngram, by_char=by_character, space_char="_"))
#     return data

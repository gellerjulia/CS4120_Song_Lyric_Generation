import nltk
import pandas as pd

nltk.download('punkt')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

def get_lyrics_in_genre(df: pd.DataFrame, genre: str, verbose: bool = False, song_limit: int = None) -> list:
	"""
	 Returns the lyrics of songs in df with the given genre.

	 Args:
			df (pandas DataFrame): dataframe of artist and lyric data
			genre (str): a music genre found in df
      verbose (bool): if True, prints the number of songs and lines
			song_limit (int): if present, the number of songs to include in the training data (used to cut down on training/generation time)

		Returns:
			A list of song lyrics, where each string is a single line in a song 

	""" 
	genre_df = df[df['genres'].apply(lambda x: genre in x)]
	songs = genre_df['lyrics'].tolist()
	num_songs_in_genre = len(songs)

	if song_limit is not None:
		songs = random.sample(songs, song_limit)

	# song lines (what we will treat as sentences) split by \n 
	song_lines = []
	for song in songs:
		song_lines.extend(song.lower().split('\n'))
        
	if verbose:
		print("Selected", len(songs), "/", num_songs_in_genre, "in the genre", genre)
		print("Total lines:", len(song_lines))
		
	return song_lines


def tokenize_line(line: str, ngram: int, 
                   by_char: bool = False, 
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
    by_char (bool):  if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = False, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool):  if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


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

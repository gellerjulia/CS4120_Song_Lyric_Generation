import pandas as pd
import random
import nltk

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
NEWLINE = "NEW"

def get_lyrics_in_genre(df: pd.DataFrame, genre: str, verbose: bool = False, by_verse: bool = False, song_limit: int = None) -> list:
	"""
	 Returns the lyrics of songs in df with the given genre.

	 Args:
			df (pandas DataFrame): dataframe of artist and lyric data
			genre (str): a music genre found in df
			verbose (bool): if True, prints the number of songs and lines
			by_verse (bool): if True, separates lyrics into full verses; if False, separates lyrics by individual lines 
			song_limit (int): if present, the number of songs to include in the training data (used to cut down on training/generation time)

		Returns:
			A list of song lyrics, where each string is a single line or verse in a song 

	""" 
	# get all song lyrics in the given genre
	genre_df = df[df['genres'].apply(lambda x: genre in x)]
	songs = genre_df['lyrics'].tolist()

	# count the number of songs (for verbose setting)
	num_songs_in_genre = len(songs)

	if song_limit is not None:
		random.seed(42) # set seed for consistent results 
		songs = random.sample(songs, song_limit)

	token_to_split = '\n\n' if by_verse else '\n' 

	# song sequences (what we will treat as sentences)
	song_seqs = []
	for song in songs:
		seqs = song.lower().split(token_to_split)

		# when splitting by verse, replace single \n with special token to preserve newlines when tokenizing 
		if by_verse:
			verses = [verse.split('\n') for verse in seqs]
			seqs = [str(" " + NEWLINE + " ").join(verse_lines) for verse_lines in verses]

		# filter out blank lines and lines with meta data
		meta_lyrics = ["chorus", "verse", "bridge", '----']
		cleaned_seqs = []
		for seq in seqs:
			if len(seq) > 0 and not any(meta in seq for meta in meta_lyrics):
				cleaned_seqs.append(seq) 

		song_seqs.extend(cleaned_seqs)
				
	if verbose:
		print("Selected", len(songs), "/", num_songs_in_genre, "in the genre", genre)
		print("Total sequences:", len(song_seqs))
		
	return song_seqs


def split_songs_into_lines(songs: list):
	"""
	Converts a list of songs into a list of individual lyric lines by splitting on newlines.
	Removes empty samples or lines with metadata to have more real lyric lines.

	Args:
		songs (list): a list of strings, where each string is a whole song

	Returns:
		song_lines (list): a list of strings, where each string is a single line in a song 
	"""
	song_lines = []
	for song in songs:
		lines = song.lower().split('\n')

		# filter out lines that are blank or have metadata 
		meta_lyrics = ["chorus", "verse", "bridge", "lyric", "----"]
		cleaned_lines = []
		for line in lines:
			if len(line) > 0 and not any(meta in line for meta in meta_lyrics):
				cleaned_lines.append(line) 

		song_lines.extend(cleaned_lines)
					
	return song_lines


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


def tokenize(data: list, ngram: int,  
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
		tokens = tokenize_line(line, ngram, sentence_begin, sentence_end)
		total += tokens
	return total

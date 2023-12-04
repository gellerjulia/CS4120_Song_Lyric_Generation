import pandas as pd
import nltk

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

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
	Tokenize each line in a list of strings and append all tokens together. 
	Glue on the appropriate number of sentence begin tokens and sentence end tokens (ngram - 1),
	except for the case when ngram == 1, when there will be one sentence begin
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

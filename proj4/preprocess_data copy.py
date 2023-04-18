'''preprocess_data.py
Preprocessing data for the recipes NLG dataset
YOUR NAMES HERE
CS343: Neural Networks
Project 4: Recurrent Neural Networks
'''
import numpy as np

def load_recipes(path):
    '''Load and the recipes dataset from https://recipenlg.cs.put.poznan.pl/. For each recipe, keep only the directions.
    
    Parameters:
    ----------
    path: the path to the file containing the dataset

    Returns:
    char_to_ix: dictionary. Map characters to int indices.
    ix_to_char: dictionary. Map int indices to characters.
    data: string corresponding to the directions from all the recipes. Separate the recipes with a newline. 
    
    TODO:
    1) Read the recipes in (use the python csv or pandas package to read the CSV)
    2) Keep the directions field for each recipe; convert this field from a list of strings to a single ascii-encoded lower-case single-line string
    3) Concatenate all the separate directions into a single big text
    4) Tokenize - we use character-level tokenization
    5) Calculate the vocabulary (sort the vocabulary alphabetically!)

    Do NOT filter stop words or do any other preprocessing for the basic project.
    '''
    pass

def sample_sequence(data, char_to_ix, length, start=-1):
    '''Sample a sequence from the data.
    
    Parameters:
    ----------
    data: string.
    char_to_ix: dictionary. Maps characters to int indices.
    length: int. Length of sequence to return.
    start: int. Where to start the sequence.

    Returns:
    xs: ndarray of type int and max length length. Int encoding of character sequence from a either start or a randomly chosen starting point in data.
    ys: ndarray of type int and max length length. Int encoding of off-by-one character sequence from xs.

    For example, if the the starting point is the first character in the third recipe and the length is 10, then:
    * xs will be an int sequence corresponding to 'n a heavy '
    * ys will be an int sequence corresponding to ' a heavy 2'
    '''
    pass
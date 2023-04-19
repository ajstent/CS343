'''preprocess_data.py
Preprocessing data for the recipes NLG dataset
YOUR NAMES HERE
CS343: Neural Networks
Project 4: Recurrent Neural Networks
'''
import numpy as np

def load_data(path):
    '''Load the dataset at path (which should be in plain text format)
    
    Parameters:
    ----------
    path: the path to the file containing a dataset in plain text format, one text per line

    Returns:
    char_to_ix: dictionary. Map characters to int indices.
    ix_to_char: dictionary. Map int indices to characters.
    data: string corresponding to all the data, with newlines separating texts. 
    
    TODO:
    1) Read the text in 
    2) Convert the list of lines (if you have one) to a single ascii-encoded lower-case string
    4) Tokenize - we use character-level tokenization
    5) Calculate the vocabulary (sort the vocabulary alphabetically!)

    Do NOT filter stop words or do any other preprocessing for the basic project.
    '''
    # read the text in
    
    # convert the list of a lines to a single lower-case string
    
    # tokenize
    
    # calculate the vocabulary
    
    # return
    
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
    xs: ndarray of type int and max length length. Int encoding of character sequence from a either start or a randomly chosen starting point in data (if start is set to -1).

    For example, if the the starting point is the first character in the third recipe in the recipe data and the length is 10, then:
    * xs will be an int sequence corresponding to 'n a heavy '
    '''
    pass
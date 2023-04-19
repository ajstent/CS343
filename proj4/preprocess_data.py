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
    2) Convert the list of lines (if you have one) to a single ascii-encoded lower-case lower-case string
    4) Tokenize - we use character-level tokenization
    5) Calculate the vocabulary (sort the vocabulary alphabetically!)

    Do NOT filter stop words or do any other preprocessing for the basic project.
    '''
    with open(path) as f:
        lines = f.readlines()
    data = ''.join(lines)
    data = data.lower()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))

    # dictionary to convert char to idx, idx to char
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return char_to_ix, ix_to_char, data
    #pass

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

    For example, if the the starting point is the first character in the third recipe in the recipe data and the length is 10, then:
    * xs will be an int sequence corresponding to 'n a heavy '
    '''
    if start < 0:
        # pick a random starting point fully within the data
        start = np.random.randint(0, len(data)-length)
    return np.array([char_to_ix[x] for x in data[start:start+length]])
    #pass
'''preprocess_data.py
Preprocessing data for the recipes NLG dataset
YOUR NAMES HERE
CS343: Neural Networks
Project 4: Recurrent Neural Networks
'''
import numpy as np
import pandas as pd
import re

np.random.seed(0)


def load_recipes(path):
    '''Load and the recipes dataset from https://recipenlg.cs.put.poznan.pl/. For each recipe, keep only the directions.
    
    Parameters:
    ----------
    path: the path to the file containing the dataset

    Returns:
    char_to_ix: dictionary. Map characters to int indices.
    ix_to_char: dictionary. Map int indices to characters.
    data: string corresponding to the directions from all the recipes. Separate the recipes with '#'. 
    
    TODO:
    1) Read the recipes in (use the python csv or pandas package to read the CSV)
    2) Keep the directions field for each recipe; convert this field from a list of strings to a single string
    3) Concatenate all the separate directions into a single big text
    4) Tokenize - we use character-level tokenization
    5) Calculate the vocabulary (sort the vocabulary alphabetically!)

    Do NOT lowercase, filter stop words, or any other preprocessing for the basic project.
    '''
    data = pd.read_csv(path, delimiter=',', encoding="ascii", encoding_errors='ignore')
    data = '#\n'.join(re.sub(r'\[|\]|\",?', '', x) for x in data['directions'])
    data = data.lower()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))

    # dictionary to convert char to idx, idx to char
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return char_to_ix, ix_to_char, data

def sample_sequence(data, char_to_ix, length, start=-1):
    '''Sample a sequence from the data.
    
    Parameters:
    ----------
    data: string.
    char_to_ix: dictionary. Maps characters to int indices.
    length: int. Length of sequence to return.
    start: int. Where to start the sequence.

    Returns:
    xs: ndarray of type int and max length length. Int encoding of character sequence from a randomly chosen starting point in data.
    ys: ndarray of type int and max length length. Int encoding of off-by-one character sequence from xs.

    For example, if the the starting point is the first character in the third recipe and the length is 10, then:
    * xs will be an int sequence corresponding to 'n a heavy '
    * ys will be an int sequence corresponding to ' a heavy 2'
    '''
    if start < 0:
        start = np.random.randint(0, len(data)-length)
    print(start)
    print(data[0:100])
    return [char_to_ix[x] for x in data[start:start+length]]
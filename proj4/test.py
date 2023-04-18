from rnn import RNN

def load_data():
        # data I/O
        data = open('input.txt', 'r').read() # should be simple plain text file

        # use set() to count the vacab size
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # dictionary to convert char to idx, idx to char
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }
        return char_to_ix, ix_to_char, data

char_to_ix, ix_to_char, data = load_data()
        
net = RNN(len(char_to_ix), 200, len(ix_to_char), 25)
net.char_to_ix = char_to_ix
net.ix_to_char = ix_to_char
net.fit(data, 
            resume_training=False, n_epochs=500, lr=1e-1, mini_batch_sz=256, verbose=2,
            print_every=100)

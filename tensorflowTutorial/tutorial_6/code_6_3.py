import tensorflow as tf
import random
import zipfile
from tutorial_6 import function_6 as f6

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = f6.load_data_jay_lyrics()
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

my_seq = list(range(30))
for X, Y in f6.data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
for X, Y in f6.data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
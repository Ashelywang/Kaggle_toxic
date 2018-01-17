import pandas as pd
import numpy as np
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

class word_collection(object):

    def __init__(self):
        self.words = set()
        self.word_len = 0


    def feed_word(self,sentence):
        try:
            word_bag = re.findall(r"[\w']+",sentence.lower())
            self.words= self.words.union(set(word_bag))
            self.word_len = len(self.words)
        except AttributeError:
            print("This sentence is \n",sentence,"\n can not be processed, will be omitted !")


    def feed_word_by_df(self,df,fname):

        length = len(df)
        print("Need to process " +  str(length) + " entries" )
        for i in range(length):
            if i != 0 and i % 10000 == 0:
                print(str(i) + "/" + str(length) +  " is finished .")
            self.feed_word(df.iloc[i][fname])

    def get_words(self):
        return self.words


def convert_text2seq(train_texts, test_texts, max_words = 100000, max_seq_len = 200, lower=True, char_level=False):

    tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=char_level)
    tokenizer.fit_on_texts(train_texts + test_texts)
    word_seq_train = tokenizer.texts_to_sequences(train_texts)
    word_seq_test = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    word_seq_train = list(sequence.pad_sequences(word_seq_train, maxlen=max_seq_len))
    word_seq_test = list(sequence.pad_sequences(word_seq_test, maxlen=max_seq_len))
    return word_seq_train, word_seq_test, word_index



def get_embedding_matrix(word_index,max_words,embed_fname):
    """
    :param word_index:
    :param max_words:
    :param embed_fname:
    :return:
    """

    file_path = '/Users/anshuwang/Documents/Work/kaggle/toxic/data/' + embed_fname
    word_set =set([key for key in word_index if word_index[key] < max_words])
    with open(file_path ,'r') as f:
        line_n = 0
        for line in f:
            if line_n == 0:
                word_num , embedding_len =  int(line.split()[0]),int(line.split()[1])
                embedding_matrix = np.zeros((max_words, embedding_len))

            else:
                line_elements  = line.split()
                tmpword = line_elements[0]
                if tmpword in word_set:
                    idx = word_index[tmpword] - 1
                    if idx < max_words:
                        for i in range(embedding_len):
                            embedding_matrix[idx,i] = float(line_elements[i+1])
                    word_set.remove(tmpword)
            line_n += 1
    return embedding_matrix,word_set












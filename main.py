import pandas as pd
import numpy as np
from importlib import reload
from embedding import fasttext
from model import lstm,train

from util import utils


train =utils.get_data("train")
test = utils.get_data("test")
max_words,max_seq_len  = 100000, 150


target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
num_classes = len(target_labels)

max_words = 100000
max_seq_len = 150


# get all the word

word_collection = fasttext.word_collection()
word_collection.feed_word_by_df(train,"comment_text")
word_collection.feed_word_by_df(test,"comment_text")

train = train.fillna("NAN")
test = test.fillna("NAN")




train['comment_seq'], test['comment_seq'], word_index = fasttext.convert_text2seq(
    train['comment_text'].tolist(), test['comment_text'].tolist(),
    max_words, max_seq_len, lower=True, char_level=False)

embedding_matrix, words_not_found = fasttext.get_embedding_matrix(word_index,max_words,"crawl-300d-2M.vec")

x = np.array(train['comment_seq'].tolist())
y = np.array(train[target_labels].values)

x_train_nn, x_test_nn, y_train_nn, y_test_nn, train_idxs, test_idxs = utils.split_data(x, y, test_size=0.2, shuffle=True, random_state=42)

test_df_seq = np.array(test['comment_seq'].tolist())


##model

lstm_model = lstm.get_lstm(embedding_matrix, num_classes, 300, max_seq_len,
                           l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.3, dense_dim=32, add_sigmoid=True)

lstm_hist = train.train(x_train_nn, y_train_nn, lstm_model, batch_size=256, num_epochs=100,
                  learning_rate=0.001, early_stopping_delta=0.0001, early_stopping_epochs=3,
                  use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=2)

y_lstm = lstm_model.predict(x_test_nn)

test_label = lstm_model.predict(test_df_seq)

train.save_predictions(test, test_label , target_labels, None)

submission = test[["id",'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']]
submission.to_csv("./result/submission.csv",index = False)


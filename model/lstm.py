import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Merge
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D

def get_lstm(embedding_matrix, num_classes, embed_dim, max_seq_len, l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.3, dense_dim=32, add_sigmoid=True):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Bidirectional(LSTM(lstm_dim, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_val))
    model.add(Dense(lstm_dim, activation="relu"))
    model.add(Dropout(dropout_val))
    model.add(Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    if add_sigmoid:
        model.add(Dense(num_classes, activation="sigmoid"))
    return model
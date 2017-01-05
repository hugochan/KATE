'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


def neural_network(input_size, n_class):
    model = Sequential()
    model.add(Dense(n_class, input_dim=input_size, init='glorot_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cv_classifier(X, Y, n_splits=10, nb_epoch=200, batch_size=10, seed=7):
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Y = np_utils.to_categorical(Y)
    estimator = KerasClassifier(build_fn=neural_network, input_size=X.shape[1], n_class=Y.shape[1], nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc = cross_val_score(estimator, X, Y, cv=kfold)
    print "accuracy: %.2f%% (%.2f%%)" % (acc.mean() * 100, acc.std() * 100)
    return acc

def classifier(X_train, Y_train, X_test, Y_test, nb_epoch=200, batch_size=10, seed=7):
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train = np_utils.to_categorical(encoder.transform(Y_train))
    Y_test = np_utils.to_categorical(encoder.transform(Y_test))
    estimator = KerasClassifier(build_fn=neural_network, input_size=X_train.shape[1], n_class=Y_train.shape[1], nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
    estimator.fit(X_train, Y_train)
    acc = estimator.score(X_test, Y_test)
    print "accuracy: %s" % acc
    return acc


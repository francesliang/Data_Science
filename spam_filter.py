# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 08:39:34 2015

@author: fliang
"""

import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

##----- Get data and data analysis
messages = [line.rstrip() for line in open('/Users/apple/Python_Projects/Data_Science/data/SMSSpamCollection')]

#for message_no, message in enumerate(messages[:10]):
#    print message_no, message
    
#messages = pandas.read_csv('C:/Temp/Python/Data_Science/data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
#                           names=["label", "message"])
                           
messages = pandas.read_csv('/Users/apple/Python_Projects/Data_Science/data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
            names=["label", "message"])

messages.groupby('label').describe()

messages['length'] = messages['message'].map(lambda text: len(text) )

#plt.hist(messages.length, bins=20)

messages.length.describe()
#print list(messages.message[messages.length > 900])

#messages.hist(column = 'length', by = 'label', bins=50)



##----- Data preprocessing
def split_into_tokens(message):
    message = unicode(message, 'utf8') # convert bytes into proper unicode
    return TextBlob(message).words
    
#print messages.message.head().apply(split_into_tokens)


def split_into_lemmas(message):
    message = unicode(message, 'utf-8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]
    
#print messages.message.head().apply(split_into_lemmas)
    
    
##----- Data to vectors
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])

#print bow_transformer.vocabulary_

message4 = messages['message'][3]
bow4 = bow_transformer.transform(message4)

message_bow = bow_transformer.transform(messages['message'])
print 'sparsity: %.2f%%' % (100.0 * message_bow.nnz / (message_bow.shape[0]*message_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(message_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4
print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

message_tfidf = tfidf_transformer.transform(message_bow)
print message_tfidf.shape

##----- Training a model and detecting spam
spam_detector = MultinomialNB().fit(message_tfidf, messages['label'])

all_predictions = spam_detector.predict(message_tfidf)

print 'accuracy', accuracy_score(messages['label'], all_predictions)
print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
print classification_report(messages['label'], all_predictions)

##----- Evaluation and experiment
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

scores = cross_val_score(pipeline, msg_train, label_train, cv=10, scoring='accuracy', n_jobs=-1)

print scores
print scores.mean(), scores.std()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, 
        train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, 
        test_scores_mean+test_scores_std, alpha=0.1, color='r')
        
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt
    
plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)
    
 
##----- Parameters tuning
params = { 'tfidf__use_idf':(True, False),
        'bow__analyzer': (split_into_lemmas, split_into_tokens)}

grid = GridSearchCV(pipeline, params, refit = True, n_jobs=-1, scoring = 'accuracy',
                    cv=StratifiedKFold(label_train, n_folds=5))
                    
nb_detector = grid.fit(msg_train, label_train)
nb_detector.grid_scores_
      
print nb_detector.predict_proba(["Hi mom, how are you?"])[0]
print nb_detector.predict_proba(["winner! credit for free"])[0]

print nb_detector.predict(["Hi mom, how are you?"])[0]
print nb_detector.predict(["winner! credit for free"])[0]

predictions = nb_detector.predict(msg_test)
print confusion_matrix(label_test, predictions)
print classification_report(label_test, predictions)


##-----  SVM classifier
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),
])
          
param_svm = [
    {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
    {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001],
    'classifier__kernel': ['rbf']},
] 

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(msg_train, label_train)
print svm_detector.grid_scores_
    
print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))  


##----- Productionalizing a predictor
with open('sms_spam_detector.pkl','wb') as fout:
    cPickle.dump(svm_detector, fout) 
    
svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))   
                 
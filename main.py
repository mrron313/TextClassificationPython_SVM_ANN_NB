from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
from sklearn import model_selection
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

print("Youtube Spam Classification")

df = pd.read_csv('Dataset/Youtube02-KatyPerry.csv', encoding='Latin-1')
df.head()

#train-test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(df['CONTENT'], df['CLASS'], test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
text_clf_nb = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])

text_clf_nb = text_clf_nb.fit(train_x, train_y)
predicted = text_clf_nb.predict(test_x)
print("\n\nNaive Bayes->",accuracy_score(predicted, test_y))
print(f1_score(test_y, predicted, average="macro"))
print(precision_score(test_y, predicted, average="macro"))
print(recall_score(test_y, predicted, average="macro"))


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
])

text_clf_svm = text_clf_svm.fit(train_x, train_y)
predicted = text_clf_svm.predict(test_x)
print("\n\nSVM->",accuracy_score(predicted, test_y))
print(f1_score(test_y, predicted, average="macro"))
print(precision_score(test_y, predicted, average="macro"))
print(recall_score(test_y, predicted, average="macro"))


from sklearn.neural_network import MLPClassifier
text_clf_ann = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-ann', MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                hidden_layer_sizes=(10, 2), random_state=1)),
])

text_clf_ann = text_clf_ann.fit(train_x, train_y)
predicted = text_clf_ann.predict(test_x)
print("\n\nANN->",accuracy_score(predicted, test_y))
print(f1_score(test_y, predicted, average="macro"))
print(precision_score(test_y, predicted, average="macro"))
print(recall_score(test_y, predicted, average="macro"))

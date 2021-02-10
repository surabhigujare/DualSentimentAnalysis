import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.classify.scikitlearn import SklearnClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.classify import ClassifierI
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import random 

short_neg = open("data1.txt","r").read()
documents1 = []

for r in short_neg.split('\n'):
    documents1.append( (r, "neg") )


short_neg_words = word_tokenize(short_neg)
tagged  = nltk.pos_tag(short_neg_words)

doc = []
na = np.array(tagged)

res = np.where((na[:,1] == 'JJ'))

res1 = np.where((na[:,1] == 'JJS'))
res2 = np.where((na[:,1] == 'JJR'))
doc1 = na[res]

doc2 = na[res1]
doc3 = na[res2]

original = []
reverse = []
feature = []
fd = open('original.txt', 'w')
fd1 = open('reverse.txt', 'w')
fd2 = open('corpus.txt','w')
for i in doc1[:,0]:
	for syn in wordnet.synsets(i):
   		for l in syn.lemmas():
        		if l.antonyms():
				original.append(syn.lemmas()[0].name())
				feature.append(syn.lemmas()[0].name())
				fd.write(syn.lemmas()[0].name())
				fd.write("\n")
				fd2.write(syn.lemmas()[0].name())
				fd2.write("\n")
				reverse.append(l.antonyms()[0].name())
				feature.append(l.antonyms()[0].name())
				fd1.write(l.antonyms()[0].name())
				fd1.write("\n")
				fd2.write(l.antonyms()[0].name())
				fd2.write("\n")
				#print "------"	
for i in doc2[:,0]:
	for syn in wordnet.synsets(i):
   		for l in syn.lemmas():
        		if l.antonyms():
				original.append(syn.lemmas()[0].name())
				feature.append(syn.lemmas()[0].name())
				fd.write(syn.lemmas()[0].name())
				fd.write("\n")
				fd2.write(syn.lemmas()[0].name())
				fd2.write("\n")
				reverse.append(l.antonyms()[0].name())
				feature.append(l.antonyms()[0].name())
				fd1.write(l.antonyms()[0].name())
				fd1.write("\n")
				fd2.write(l.antonyms()[0].name())
				fd2.write("\n")
				#print "------"
for i in doc3[:,0]:
	for syn in wordnet.synsets(i):
   		for l in syn.lemmas():
        		if l.antonyms():
				original.append(syn.lemmas()[0].name())
				feature.append(syn.lemmas()[0].name())
				fd.write(syn.lemmas()[0].name())
				fd.write("\n")
				fd2.write(syn.lemmas()[0].name())
				fd2.write("\n")
				reverse.append(l.antonyms()[0].name())
				feature.append(l.antonyms()[0].name())
				fd1.write(l.antonyms()[0].name())
				fd1.write("\n")
				fd2.write(l.antonyms()[0].name())
				fd2.write("\n")
				#print "------"
fd.close()
fd1.close()
fd2.close()
o = open("original.txt","r").read()
r = open("reverse.txt","r").read()
documents2 = []
documents3 = []

for p in o.split('\n'):
    documents2.append( (p, "neg") )
for  q in r.split('\n'):
    documents3.append( (q, "pos") )

#print documents2
#print documents3

final = []
print len(documents2)
print "Original Words labeled as negative:"
print documents2
print "============================================================================================="
print len(documents3)
print "Reversed Words labeled as positive:"
print documents3
while True:
    try:
        final.append(documents2.pop(0))
        final.append(documents3.pop(0))
    except IndexError:
        break

print len(final)
feature1 = nltk.FreqDist(feature)
word_features = list(feature1.keys())[:470]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in final]

testing_set = featuresets[100:]
training_set = featuresets[:416]
testing_list = feature[100:]
training_list = feature[:416]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(20)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


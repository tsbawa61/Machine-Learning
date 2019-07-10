import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(','not',"stupendous" ,'hopeless'] #,,
 
positive_features = [(word_feats(positive_vocab), 'pos')]
negative_features = [(word_feats(negative_vocab), 'neg')]

print("==== positive_features :\n",positive_features)
print("==== negative_features :\n",negative_features)

train_set = negative_features + positive_features

print("==== train_set:\n",train_set)

print("==== Now Training train_set with NaiveBayesClassifier.train(train_set) statement")

classifier = NaiveBayesClassifier.train(train_set) 
print("==== Trained train_set with NaiveBayesClassifier as per above statement")

 
# Predict
#sentence = "Awesome movie I liked it"
#sentence = "Terrible movie I hate it"
#sentence = "Bad movie"
#sentence = "Stupendous movie"
sentence = "Hopeless movie"

sentence = sentence.lower()
words = sentence.split(' ')
print("======words: \n",words)
print(classifier.classify( word_feats(words)))





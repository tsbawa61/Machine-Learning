
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
# movie reviews folder: C:\Users\Admin\AppData\Roaming\nltk_data\corpora\movie_reviews

def word_feats(words):
    return dict([(word, True) for word in words])

print(word_feats(["This", "moview", "is", "Awesome"]))
      
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
print(negids[:5])

str=movie_reviews.words(fileids='neg/cv000_29416.txt')

print("str=", str)

input("Press Enter to continue")
      
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
print(negfeats[:5])
input("Press Enter")
print(posfeats[:5])
 
negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)
print(negcutoff,poscutoff)
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

#Testing our own movie reviews
test_data_features=word_feats(movie_reviews.words(fileids='../test/kabir_review.txt'))
test_data_features1=word_feats(movie_reviews.words(fileids='../test/tube_review.txt'))
print("***Reviews:")
print (classifier.classify(test_data_features))
print (classifier.classify(test_data_features1))




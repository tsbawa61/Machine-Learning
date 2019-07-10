import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = [] 

for w in movie_reviews.words():
    all_words.append(w.lower())

#The FreqDist class is used to encode “frequency distributions”, which count the number of times that 
#each outcome of an experiment occurs.
#Formally, a frequency distribution can be defined as a function mapping from each sample to the number of times 
#that sample occurred as an outcome.
#fdist = FreqDist(word.lower() for word in word_tokenize(sent)

all_words = nltk.FreqDist(all_words)
print(all_words.B(),',',all_words.freq('plot'),',',all_words.max(),',',all_words.freq(','))
print(all_words.most_common(15))
print(all_words["plot"])
abc=all_words.keys()
print(type(abc),len(abc))
print(list(abc)[:20])

#word_features contains the top 3,000 most common words. 
word_features = list(all_words.keys())[:3000]

#The function below will find these top 3,000 words in our positive and negative documents, 
#marking their presence as either positive or negative:

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#As in below, For all of our documents, we save the feature existence booleans and 
#their respective positive or negative categories :

featuresets = [(find_features(rev), category) for (rev, category) in documents]
print("len(featuresets) : ",len(featuresets))

# set that we'll train our classifier with
training_set = featuresets[:1600]
print("len(training_set) : ",len(training_set))

# set that we'll test against.
testing_set = featuresets[1600:]
print("len(testing_set) : ",len(testing_set))

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

nltk.NaiveBayesClassifier.classify(testing_set[0])
classifier.show_most_informative_features(15)


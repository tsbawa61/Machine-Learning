import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])

print("len(names):",len(names))
tmp = input("Press Any Key: ")

print("names",names[:10])
tmp = input("Press Any Key: ")

def gender_features(word): 
    return {'last_letter': word[-1]}
 
print(gender_features("Caesar") )
print(gender_features("Jolie"))
tmp = input("Press Any Key: ")

featuresets = [(gender_features(n), g) for (n,g) in names]
print("len(featuresets):",len(featuresets))
tmp = input("Press Any Key: ")

print("featuresets",featuresets[:10])

train_set = featuresets[100:]
test_set =  featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('Caesar')))
print(classifier.classify(gender_features('Jolie')))

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)
print( "classifier:",classifier)
# Predict
print(classifier.classify(gender_features('Frank')))
print(classifier.classify(gender_features('Cleopetra')))
print(classifier.classify(gender_features('Romeo')))
print(classifier.classify(gender_features('Juliet')))



# Predict
_="""name = input("Name: ")
print(classifier.classify(gender_features(name)))

name = input("Another Name: ")
print(classifier.classify(gender_features(name)))

name = input("Another Name2: ")
print(classifier.classify(gender_features(name)))"""


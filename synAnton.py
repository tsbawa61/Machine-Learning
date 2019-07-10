from bs4 import BeautifulSoup 
import urllib.request 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


syn = wordnet.synsets("pain")
print(syn)

print(syn[0].definition())
input("Press any key to continue")
print(syn[0].examples())
input("Press any key to continue")

syn = wordnet.synsets("NLP")
print(syn[0].definition())
syn = wordnet.synsets("Python")
print(syn[0].definition())
input("Press any key to continue")

synonyms = []
for syn in wordnet.synsets('Computer'):
    print(syn.lemmas())
    input("Press any key to continue")
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)
input("Press any key to continue")

antonyms = []
for syn in wordnet.synsets("small"):
    print(syn.lemmas)
    input("Press any key to continue")
    for l in syn.lemmas():
#        print(l)
#        input("Press any key to continue")
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)
input("Press any key to continue")

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))
input("Press any key to continue")

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))


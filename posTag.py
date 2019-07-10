import nltk
from nltk.tokenize import word_tokenize

text = word_tokenize("Smart John of Washington met sweet Jenny who was also very beautiful and was going quickly to her big home")

# We will get words tagged as Nouns, Verbs, Etc by the following statement
pos_text=nltk.pos_tag(text)

print(pos_text)

print("Nouns in the sentence are :")
for (word,typ) in pos_text:
    if typ=='NNP' :
        print(word)
        
print("Verb in the sentence are :")
for (word,typ) in pos_text:
    if typ=='VBD' :
        print(word)

print("Adjective in the sentence are :")
for (word,typ) in pos_text:
    if typ=='RB' :
        print(word)
  

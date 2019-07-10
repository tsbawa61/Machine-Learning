import nltk 
from nltk.tokenize import sent_tokenize

mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude." 
print(sent_tokenize(mytext))

mytext = "Hello Mr. Adam, how are you? I hope everything is going well! Today is a good day, see you dude."
print(sent_tokenize(mytext))

from nltk.tokenize import word_tokenize
mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
print(word_tokenize(mytext))

mytext = "Hello Mr. Adam, how are you? Are you working in U.N.O?"
print(word_tokenize(mytext))

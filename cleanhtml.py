from bs4 import BeautifulSoup
#Beautiful Soup is a Python library for pulling data out of HTML and XML files. 

import urllib.request 
import nltk 
from nltk.corpus import stopwords

file = open(r"D:\backupKRK2206\PythonProgs\cleanhtml1.txt","w") 

response = urllib.request.urlopen(r'http://php.net/') 
html = response.read()
print(html)

#input("Press Enter to continue")
#file.write ("html:\n"+html.decode())

soup = BeautifulSoup(html,"html5lib") 
#The following will give text containing a and b tags
txt=soup.find_all(["a", "b"])
print (txt)


text = soup.get_text()
#You can tell Beautiful Soup to strip whitespace from the beginning and end of each bit of text:
text = soup.get_text(strip=True)
print("text:\n"+text)
file.write ("text:\n"+text)

tokens = [t for t in text.split()]
print("tokens:", tokens)
input("Press Enter to continue")

clean_tokens = tokens[:] 

sr = stopwords.words('english')
print("Stop Words:", sr)
input("Press Enter to continue")

for token in tokens: 
    if token in stopwords.words('english'): 
        clean_tokens.remove(token)
        
freq = nltk.FreqDist(clean_tokens)
print("freq:", freq)
input("Press Enter to continue")

print("freq.items():", freq.items())
input("Press Enter to exit freq.items() ")

l=[]
for key,val in freq.items():
    t=[]
    t.append(key)
    t.append(val)
    #print (str(key) + ':' + str(val))
    l.append(tuple(t))
        
print(l)
# take second element for sort

def takeSecond(elem):
    return elem[1]

# sort list with key
l.sort(key=takeSecond,reverse=True)
print("===============")
print(l)

file.close() 

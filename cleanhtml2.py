from bs4 import BeautifulSoup 
import urllib.request 
import nltk 
from nltk.corpus import stopwords

file = open("F:\\pythonProgs\\cleanhtml1.txt","w") 


response = urllib.request.urlopen('http://php.net/') 
html = response.read()
print(html)
input("Press Enter to continue")
#file.write ("html:\n"+html.decode())

soup = BeautifulSoup(html,"html5lib") 
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

for key,val in freq.items():
   print (str(key) + ':' + str(val))

file.close() 

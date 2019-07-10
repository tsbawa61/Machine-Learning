import json

# read file
myfile=open(r'D:\backupKRK2206\PythonProgs\example.json', 'r')
#In the above use your own path of file

data=myfile.read()

# parse file
obj = json.loads(data)
print(type(obj))
print(obj)

print(type(obj['office']))
print(obj['office'],'\n')

print(type(obj['office']['medical']))
print(obj['office']['medical'],'\n')

for i in range(len(obj['office']['medical'])):
    print(obj['office']['medical'][i]['room-number'])
    
print(type(obj['office']['parking']))
print(obj['office']['parking'],'\n')

for kee in obj['office']['parking']:
    print(kee,'->', obj['office']['parking'][kee]) 

"""
# show values
print("usd: " + str(obj['usd']))
print("eur: " + str(obj['eur']))
print("gbp: " + str(obj['gbp']))
"""

file = open("F:\\pythonProgs\\mlsimple.txt","w") 

from random import randint
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()

for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2*b) + (3*c)
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

print(TRAIN_INPUT)
input("Press Enter key to continue-1")

file.write ("\nTRAIN_OUTPUT:\n")
file.writelines("%s," % x for x in TRAIN_INPUT )

print(TRAIN_OUTPUT)
input("Press Enter key to continue-2")
file.write ("\nTRAIN_OUTPUT:\n")
file.writelines("%s," % x for x in TRAIN_OUTPUT )


#Training the LinearRegression Classifier
from sklearn.linear_model import LinearRegression

#Training the LinearRegression Classifier
predictor = LinearRegression(n_jobs=-1) #n_jobs = int, optional, default is 1 
#The number of jobs to use for the computation. If -1 all CPUs are used

predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT) #Fit linear model.

X_TEST = [[20, 40, 60]]
outcome = predictor.predict(X=X_TEST) #Predict using the linear model
print('\n Outcome : ', outcome) #Output: Outcome :  [140.]

coefficients = predictor.coef_ #Estimated coefficients for the linear regression problem. 

print('\n Coefficients : ', coefficients) #Output: Coefficients :  [1. 2. 3.]

file.close() 

#!/usr/bin/env python
from sklearn.neural_network import MLPClassifier
import numpy as np

f=open('output.txt','w')
#load training data file
x= np.array(np.loadtxt('4train_x.txt',delimiter=','))
y = np.array(np.loadtxt('4train_y.txt',delimiter=','))

model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict(np.loadtxt('4test.txt',delimiter=','))
for row in predicted:
	s = str(row)	
	f.write(s+'\n')

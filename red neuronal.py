# neural network with keras tutorial
import numpy as np #To perform operations with arrays
import keras #to Neural Network
from keras.utils import to_categorical #to classifcation neural network
from keras.models import Sequential #to define layers
from keras.layers import Dense #to define full conected layers
import matplotlib.pyplot as plt #to plot
import pandas as pd #to read excel

train = pd.read_excel (r'Rq_4beam.xlsx') #Read file with input data (traffic demand on each beam)
labels = pd.read_excel (r'Label-4 beams.xlsx') #Read the file with the corresponding resource configuration label 
labels= labels-1 #The labels go from 1 to N, where N is the number of configurations, in python you start from zero, so the labels must go from 0 to N-1.
onehot = to_categorical(labels, num_classes=len(np.unique(labels))) #The labels are transformed into vectors with zeros and ones.

# in this example
np.random.seed(100)
indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0]*0.1)
test_idx, training_idx = indices[:valid_cnt],\
                indices[valid_cnt:]
test, train = train.iloc[test_idx,:],\
                train.iloc[training_idx]
onehot_test, onehot_train = onehot[test_idx,:],\
                onehot[training_idx,:]

print('Forma de datos de Entrenamiento:',train.shape,'\nForma de datos del test:',test.shape)


from keras.models import Sequential, save_model, load_model
from keras.layers import Dropout, Dense

model = Sequential() #A sequential model is defined
input_size = 4 # Number of neurons to the input layer, equals number of beams
output_size = 256 # Number of neurons to the output layer, equals Number of possible configurations

model.add(Dense(32,input_dim=input_size,activation = 'relu')) # A full connected layer with 32 neurons, being the first hidden layer, the size of the input layer must be set, an activation function ReLu is used.
model.add(Dense(256,activation = 'relu')) #A full connected layer with 256 neurons, uses the ReLu activation function.
model.add(Dropout(0.5)) #Helps to avoid overfitting
model.add(Dense(500,activation = 'relu')) #A full connected layer with 500 neurons, uses the ReLu activation function.
model.add(Dropout(0.5)) #Helps to avoid overfitting

model.add(Dense(output_size,activation='softmax')) # This is the output layer where the classification is done, with N neurons, where N is the number of configurations, sofmax is used to obtain probabilities. 
model.summary() #Displays the network architecture


from keras.optimizers import SGD
# loss is cost fuction, in this case is selected "ctaegorical crossentropy"
# SGD is Gradient descent (with momentum) optimizer.
#lr is the learning rate, represents how fast the search for optimization parameters is performed.
#decay - you can set a decay function for the learning rate. This will adjust the learning rate as training progresses.
#momentum - accelerates SGD in the relevant direction and dampens oscillations. Basically it helps SGD push past local optima, gaining faster convergence and less oscillation. A typical choice of momentum is between 0.5 to 0.9.
#Nesterov is a different version of the momentum method which has stronger theoretical converge guarantees for convex functions. In practice, it works slightly better than standard momentum

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])


# model.fit to train the Network
#first select the training data: train,onehot_train
# Number of epochs
# size of batch
#select the test data: test,onehot_test
# verbose to see in real time the training
history=model.fit(train,onehot_train,
                 epochs=2,
                 batch_size=500,
                 validation_data=(test,onehot_test),
                 verbose=1)


plt.plot(history.history['accuracy'],'bo')
plt.plot(history.history['val_accuracy'],'rX')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.grid()
plt.show()

plt.plot(history.history['loss'],'bo')
plt.plot(history.history['val_loss'],'rX')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.grid()
plt.show()


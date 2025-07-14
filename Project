# Important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv("Credit_Card_Application.csv")

# Creating features
y = data.iloc[: -1].values 
x = data.iloc[:, :-1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaller = MinMaxScaler(feature_range = (0, 1))
x = scaller.fit_transform(x)


# Now. I'm starting training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=0, learning_rate=0.5) 
som.random_weights_init(x)
som.train_random(x, num_iterations=85) 


# We have to visualize our results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) 
colorbar()
markers = ['o', 's'] 
colors = ['b', 'r'].
for i, x in enumerate(x):
  winning_node = som.winner(x)
  plot(winning_node[0] + 0.5, winning_node[1] + 0.5,
       markers[y[i]],
       markeredgecolor = color[y[i]],
       markerfacecolor = "None",
       markersize = 10,
       markeredgewidth = 2)
show()


# Identifies the frauds
mappings = som.win_map(x) 
frauds = np.concatenate((mappings[(5, 3)], mappings[(8, 3)]), axis=0)
frauds = scaller.inverse_transform(frauds)

# Matrix of features, include relevant columns 
customer = dataset.iloc[:, 1:].values # Include all the columns except the last one. ~ values (Create numpy array)

# Dependend variable! Binary outcome (0 if there is no fraud, 1 if there is a fraud)
# Initialise variables of all items (690)
# Replace 1 if there is fraud.
# Use numpy (zeros) to get 0

frauds_list = np.zeros(len(data))


# Now I'm creating a loop to check if the customer belongs to the frauds list (Then replace 0 -> 1)
for i in range(len(data)):
  if data.iloc[i, 0] in frauds_list: # 'i' in iloc corespond to lines (customers) of the data and 0 becouse the first column corespond to the customer id 
    frauds_list[i] = 1



# Create our second part (+ ANN structure)




from sklearn.preprocessing import StandardScaler
scaller = StandardScaler()
customers = scaller.fit_transform(customers) # it will transform our data in numbers instead letters etc

# Import Keras Libraries
from keras.model import Sequential
from keras.layers import Dense

# Creating ANN
classifier = Sequential()

# Get first input layers and first hidden layers
classifier.add(Dense(units=4, kernel_initializer="uniform", activation="relu", input_dim=15)) # 'input_dimensionm = number of features 


# Make output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid")) # sigmoid function!


# Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# fiting the ANN
classifier.fit(customers, frauds_list, batch_size=1, epochs=2) #Epochs 1/2 will be enough (Small amount of dataset)

# predict the frauds propability
predictor = classifier.predict(customers)


predictor = np.concatenate((data.iloc[:, 0:1].values,  predictor), axis=1))
# Data iloc (All the customers id), Two dimensional array so you have to change also to 2d array + .values to make it numpy array value
# axis = 0 (Vertical concatenation) axis = 1 (Horizontal concatenation)

# Sorted values from the highest to lowest
predictor = predictor[predictor[:, 1].argslrf()]

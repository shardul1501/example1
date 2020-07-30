# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('AI-DataTrain.csv')
dataset.drop(columns = ['Unnamed: 0'], inplace = True)
dataset_test = pd.read_csv('AI-DataTest.csv')

def preprocess(dataset):
    # Transposing the dataset
    data_trans = dataset.T
    data_trans.head()
    
    # Calculating the Response Accuracy
    sums = data_trans.sum(axis = 1)
    response_accuracy = sums/len(list(data_trans.columns))
    response_accuracy
    
    # Calculating Question Difficulty
    diff_level = 1 - response_accuracy

    # Adding both lists to dataset
    data_trans['ResponseAccuracy'] = response_accuracy
    data_trans['QuestionDifficulty'] = diff_level
    
    return data_trans

def klearn(dataset):
    from sklearn.cluster import KMeans
    # Preparing the training data
    X = dataset.iloc[:, -1].values
    X = X.reshape(-1,1)
    
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    
    # Making a classes column in dataset
    classes = list(y_kmeans)
    dataset['Classes'] = classes
    return dataset

def classify(training, test_data):
    
    # Preparing the training data
	X_train = training.iloc[:, -3:-1].values
	y = training.iloc[:, -1].values
	y_train = np.array(y,dtype='int32')
    
    # Preparing the test data
	X_test = test_data.iloc[:, -2:].values

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	# Importing the Keras libraries and packages
	import keras
	from keras.models import Sequential
	from keras.layers import Dense

	# Initialising the ANN
	classifier = Sequential()

	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

	# Adding the second hidden layer
	classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'softmax'))

	# Adding the output layer
	classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))

	# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

	# Fitting the ANN to the Training set
	classifier.fit(X_train, y_train, batch_size = 5, epochs = 500)

	# Making predictions and evaluating the model
	'''losses = pd.DataFrame(classifier.history.history)
	losses.plot()
	plt.show()'''

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	output = []
	for y in y_pred:
	    ans = np.where(y == max(y))[0][0]
	    output.append(ans)

	# Saving the model
	#classifier.save("model.h5")

	# Generating the output
	df ={'Questions':list(test_data.index), 'Weights':output}
	output_df = pd.DataFrame.from_dict(df)
	output_df.to_excel('output.xls')


# Preprocessing the data
traindata = preprocess(dataset)
test_data = preprocess(dataset_test)

# Clustering the data
training = klearn(traindata)

# Classifying the test data into classes
classify(training, test_data)
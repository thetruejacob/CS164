import numpy as np

def get_int_data(csvfile,split=.5,gs=False,seed=0):
	"""
	Get data from a csv file.
	
	INPUTS:
	csvfile (required): a string '_____.csv' indicating where to read from - see README for how the .csv should be formatted
	
	Remaining inputs are optional:
	split: (default = 0.5) a number between 0 and 1 indicating the approximate percentage of the dataset to be parsed into a training set
	gs: (default = False) indicates whether or not the dataset has included group structure (again, see README)
	seed: (default = 0) random seed for random split of train/test set

	OUTPUTS:
	trainset: a training set of the correct size split*numSamples
	trainlabels: labels corresponding to the datapoints in trainset
	testset: the complement of trainset in the dataset
	testlabels: labels corresponding to the datapoints in testset
	group_structure: an array describing the group structure, if gs=True (otherwise, returns empty array)
	
	"""
	A = np.genfromtxt(csvfile,delimiter=',')
	numFeatures = len(A[0]) - 1

	numSamples = len(A)
	I = A
	group_structure = [] 

	# get group structure:
	if gs:
		group_structure = I[numSamples-1][:numFeatures]
		I = I[:numSamples-1]

	np.random.seed(seed)	
	mask = np.random.choice([False,True],len(I),p=[1-split,split])
	
	train = I[mask]
	test = I[~mask]
	trainset = train[:,:numFeatures]
	trainlabels = train[:,numFeatures]
	testset = test[:,:numFeatures]
	testlabels = test[:,numFeatures]
	
	return trainset, trainlabels, testset, testlabels, group_structure


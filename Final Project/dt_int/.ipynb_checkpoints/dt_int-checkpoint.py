from __future__ import print_function
from node import *
from get_int_data import *
import cplex
import numpy as np
import csv
import sys
import random
from copy import deepcopy

def depth_k_tree(k):
	"""Generates a balanced decision tree of depth k."""

	tree = Tree()
	tree.add_node(0)
	count = 1
	for j in range(k-1):
		leaves = tree.get_leaf_nodes()
		for leaf in leaves:
			tree.add_node(count,leaf,'L')
			count = count + 1
			tree.add_node(count,leaf,'R')
			count = count + 1

	return tree

def build_int_model(tree,I,labels,C=1,maxtime=1800,priorities=False,anchor=True,deleted=True,relaxed=True,relaxedobj=True,strengthen=True,gs = []):
	"""
	Sets up the IP model to solve for an ODT and then solves it.
	
	INPUTS:
	tree (required): a tree object from the Tree class in node.py. 
	I (required): n (samples) x p (features) numpy array of training data
	labels (required): length n (samples)) numpy array of 0-1 valued class labels
	
	Remaining inputs are optional:
	C: (default 1) weight on negative-class examples (set less than 1 to emphasize positive-class examples)
	maxtime: (default 1800) number of seconds before CPLEX quits and returns best found integer solution
	priorities: (default False) tell CPLEX priorities on which to branch variables (v > z > anything else,
	with v variables closest to the root node being the most important) 
	anchor: (default True) use the anchoring (symmetry-breaking) equalities discussed in the write-up 	
	deleted: (default True) delete spurious variables (this would be done by CPLEX automatically, but might as well)
	relaxed: (default True) relax the integrality of the z variables adjacent to leaf nodes
	relaxedobj: (default True) relax the integrality of the correct classification variables
	strengthen: (default True) use strengthened inequalities for paths through the tree
	gs: (default []) provide the group structure of the instance - see the README for more information

	OUTPUT:
	will be written on the tree object

	"""	
	p = cplex.Cplex()
	p.objective.set_sense(p.objective.sense.maximize)
	
	numSamples = len(I)
	numFeatures = len(I[0])

	# get group structure from inputs, or else squeeze it out	
	if gs != []:
		groups = gs
		numGroups = int(max(groups)+1)
	else:
		# clone
		I2 = np.zeros((numSamples,2*numFeatures))
		for j in range(numFeatures):
			I2[:,2*j] = I[:,j]
			I2[:,2*j+1] = 1-I[:,j]
		numGroups = numFeatures 
		I = I2
		groups = np.zeros(2*numFeatures)
		for j in range(numGroups):
			groups[2*j] = j
			groups[2*j+1] = j
		numFeatures = 2*numFeatures

	leaves = tree.get_leaf_nodes()
	numNodes = max(leaves) + 1

	# set up the buckets
	numBuckets = 0
	bucket_dict = {}
	for leaf in leaves:
		bucket_dict[leaf] = numBuckets
		numBuckets += 1
	
	# establish the variables
	names = []
	for k in range(numNodes):
		for j in range(numFeatures):
			name = 'z_'+repr(j)+'_'+repr(k)
			names.append(name)
	z_ = numNodes*numFeatures
	
	for k in range(numNodes):
		for g in range(numGroups):
			name = 'v_'+repr(g)+'_'+repr(k)
			names.append(name)
	zv_ = z_ + numNodes*numGroups

	if deleted:
		numGood = np.count_nonzero(labels)
		numBad = numSamples-numGood	
		for b in range(numBuckets):
			for i in range(numBad):
				name = 'cbad_'+repr(i)+'_'+repr(b)
				names.append(name)
		for b in range(numBuckets,2*numBuckets):
			for i in range(numGood):
				name = 'cgood_'+repr(i)+'_'+repr(b)
				names.append(name)
		zvc_ = zv_ + numBuckets*numSamples
	else:
		for b in range(2*numBuckets):
			for i in range(numSamples):
				name = 'c_'+repr(i)+'_'+repr(b)
				names.append(name)
		zvc_ = zv_ + 2*numBuckets*numSamples

	# path enumeration
	paths = []
	senses = []
	for leaf in leaves:
		curr = leaf
		path = []
		path.append(curr)
		leafsense = []
		while curr != 0:
			leafsense.append(tree[curr].sense)
			curr = tree[curr].parent
			path.append(curr)
		paths.append(path)
		senses.append(leafsense)
	
	if strengthen:
		A = np.zeros((numNodes*((numNodes+2)*numSamples+numGroups*numFeatures+2),zvc_))
	else:
		A = np.zeros((numNodes*((numNodes+1)*numSamples + numGroups*numFeatures+2)+1,zvc_))
	rhs = []
	ineq = ''
	constraint_cnt = 0
	cnames = []

	# pick a group at each node
	for k in range(numNodes):
		for g in range(numGroups):
			A[constraint_cnt,z_ + k*numGroups + g] = 1
		rhs.append(1)
		ineq = ineq + "E"
		constraint_cnt = constraint_cnt + 1
		cname = 'OneGroupPerNode_Node_'+repr(k)
		cnames.append(cname)

	# group hierarchy constraints:
	group_no = 0
	curr_feature = 0
	for j in range(numFeatures):
		ind = groups[j]
		if ind != group_no or j == (numFeatures - 1):
			prev_feature = curr_feature
			if j == (numFeatures - 1):
				curr_feature = numFeatures
			else:
				curr_feature = j
			for k in range(numNodes):
				for j in range(prev_feature,curr_feature):
					A[constraint_cnt,z_ + k*numGroups + group_no] = -1
					A[constraint_cnt,k*numFeatures + j] = 1

					rhs.append(0)
					if anchor:
						if j == prev_feature and k not in leaves: #and (j+1) < curr_feature:
							ineq = ineq + "E"
						else:
							ineq = ineq + "L"
					else:
						ineq = ineq + "L"
					constraint_cnt += 1
					cname = 'GroupHierarchy_Group_'+repr(group_no)+'_Feature_'+repr(j)+'_Node_'+repr(k)
					cnames.append(cname)
			group_no = ind

	if strengthen:
		# tree structure constraints (left)
		for k in range(numNodes):
			sensescopy = deepcopy(senses)
			for sense in sensescopy:
				sense.insert(0,'L')
			collected_buckets = []
			for path in range(len(paths)):
				if k in paths[path]:
					ind = paths[path].index(k)
					if sensescopy[path][ind] == 'L':
						collected_buckets.append(bucket_dict[paths[path][0]])	
					if sensescopy[path][ind] == 'L' and ind != 0:
						collected_buckets.append(numBuckets + bucket_dict[paths[path][0]])
			if deleted:
				badindex = 0
				goodindex = 0
				for i in range(numSamples):
					if labels[i] == 0:
						for j in range(numFeatures):
							A[constraint_cnt,k*numFeatures + j] = -I[i][j]
						for c in collected_buckets:
								if c < numBuckets:
									A[constraint_cnt,zv_ + c*numBad + badindex] = 1
						rhs.append(0)
						ineq = ineq + "L"
						constraint_cnt = constraint_cnt + 1
						cname = 'BadAggregateConstraintLeft_'+repr(k)+'_Samples_'+repr(i)
						cnames.append(cname)
						badindex += 1
					else:
						if k not in leaves:
							for j in range(numFeatures):
								A[constraint_cnt,k*numFeatures + j] = -I[i][j]
							for c in collected_buckets:
									if c >= numBuckets:
										A[constraint_cnt,zv_ + numBuckets*numBad + (c-numBuckets)*numGood + goodindex] = 1
							rhs.append(0)
							ineq = ineq + "L"
							constraint_cnt = constraint_cnt + 1
							cname = 'GoodAggregateConstraintLeft_'+repr(k)+'_Samples_'+repr(i)
							cnames.append(cname)
							goodindex += 1
							
			else:
				for i in range(numSamples):
					for j in range(numFeatures):
						A[constraint_cnt,k*numFeatures + j] = -I[i][j]
					for c in collected_buckets:
						A[constraint_cnt,zv_ + c*numSamples + i] = 1
					rhs.append(0)
					ineq = ineq + "L"
					constraint_cnt = constraint_cnt + 1
					cname = 'AggregateConstraintLeft_'+repr(k)+'_Samples_'+repr(i)
					cnames.append(cname)

		# tree structure constraints (right)
		for k in range(numNodes):
			sensescopy = deepcopy(senses)
			for sense in sensescopy:
				sense.insert(0,'R')
			collected_buckets = []
			for path in range(len(paths)):
				if k in paths[path]:
					ind = paths[path].index(k)
					if sensescopy[path][ind] == 'R':
						collected_buckets.append(numBuckets + bucket_dict[paths[path][0]])	
					if sensescopy[path][ind] == 'R' and ind != 0:
						collected_buckets.append(bucket_dict[paths[path][0]])
			if deleted:
				goodindex = 0
				badindex = 0
				for i in range(numSamples):
					if labels[i] == 1:
						for j in range(numFeatures):
							A[constraint_cnt,k*numFeatures + j] = I[i][j]
						for c in collected_buckets:
							if c >= numBuckets:
								A[constraint_cnt,zv_ + numBuckets*numBad + (c-numBuckets)*numGood + goodindex] = 1
						rhs.append(1)
						ineq = ineq + "L"
						constraint_cnt = constraint_cnt + 1
						cname = 'GoodAggregateConstraintRight_'+repr(k)+'_Samples_'+repr(i)
						cnames.append(cname)
						goodindex += 1
					else:
						if k not in leaves:
							for j in range(numFeatures):
								A[constraint_cnt,k*numFeatures + j] = I[i][j]
							for c in collected_buckets:
								if c < numBuckets:
									A[constraint_cnt,zv_ + c*numBad + badindex] = 1
							rhs.append(1)
							ineq = ineq + "L"
							constraint_cnt = constraint_cnt + 1
							cname = 'BadAggregateConstraintRight_'+repr(k)+'_Samples_'+repr(i)
							cnames.append(cname)
							badindex += 1
						
							
			else:
				for i in range(numSamples):
					for j in range(numFeatures):
						A[constraint_cnt,k*numFeatures + j] = I[i][j]
					for c in collected_buckets:
						A[constraint_cnt,zv_ + c*numSamples + i] = 1
					rhs.append(1)
					ineq = ineq + "L"
					constraint_cnt = constraint_cnt + 1
					cname = 'AggregateConstraintRight_'+repr(k)+'_Samples_'+repr(i)
					cnames.append(cname)
	else:
		# weaker left tree constraints
		for k in range(numNodes):
			collected_buckets = []
			sensescopy = deepcopy(senses)
			for sense in sensescopy:
				sense.insert(0,'L')
			for path in range(len(paths)):
				if k in paths[path]:
					ind = paths[path].index(k)
					if sensescopy[path][ind] == 'L':
						collected_buckets.append(bucket_dict[paths[path][0]])	
					if sensescopy[path][ind] == 'L' and ind != 0:
						collected_buckets.append(numBuckets + bucket_dict[paths[path][0]])
			if deleted:
				goodindex = 0
				badindex = 0
				for i in range(numSamples):
					if labels[i] == 0:
						for c in collected_buckets:
							if c < numBuckets:
								for j in range(numFeatures):
									A[constraint_cnt,k*numFeatures + j] = -I[i][j]
								A[constraint_cnt,zv_ + c*numBad + badindex] = 1
								rhs.append(0)
								ineq = ineq + "L"
								constraint_cnt = constraint_cnt + 1
								cname = 'BadAggregateConstraintLeft_'+repr(k)+'_Samples_'+repr(i)
								cnames.append(cname)
						if k == 0: #just do this once
							for c in range(numBuckets):
								A[constraint_cnt,zv_ + c*numBad + badindex] = 1
							rhs.append(1)
							ineq = ineq + "L"
							constraint_cnt += 1
							cname = 'OneBucketPerSample_Sample_' + repr(i) + '_Bucket_'+repr(c)
							cnames.append(cname)
						badindex += 1
					else:
						for c in collected_buckets:
							if c >= numBuckets:
								for j in range(numFeatures):
									A[constraint_cnt,k*numFeatures + j] = -I[i][j]
								A[constraint_cnt,zv_ + numBuckets*numBad + (c-numBuckets)*numGood + goodindex] = 1
								rhs.append(0)
								ineq = ineq + "L"
								constraint_cnt = constraint_cnt + 1
								cname = 'GoodAggregateConstraintLeft_'+repr(k)+'_Samples_'+repr(i)
								cnames.append(cname)
						if k == 0: #just do this once
							for c in range(numBuckets):
								A[constraint_cnt,zv_ + numBuckets*numBad + c*numGood + goodindex] = 1
							rhs.append(1)
							ineq = ineq + "L"
							constraint_cnt += 1
							cname = 'OneBucketPerSample_Sample_' + repr(i) + '_Bucket_'+repr(c)
							cnames.append(cname)
						goodindex += 1
						
			else:
				for i in range(numSamples):
					for c in collected_buckets:
						for j in range(numFeatures):
							A[constraint_cnt,k*numFeatures + j] = -I[i][j]
						A[constraint_cnt,zv_ + c*numSamples + i] = 1
						rhs.append(0)				
						ineq = ineq + "L"
						constraint_cnt = constraint_cnt + 1
						cname = 'BucketConstraintLeft_'+repr(k)+'_Sample_'+repr(i)+'_Bucket_'+repr(c)
						cnames.append(cname)
		# weaker right tree constraints
		for k in range(numNodes):
			collected_buckets = []
			sensescopy = deepcopy(senses)
			for sense in sensescopy:
				sense.insert(0,'R')
			for path in range(len(paths)):
				if k in paths[path]:
					ind = paths[path].index(k)
					if sensescopy[path][ind] == 'R':
						collected_buckets.append(numBuckets + bucket_dict[paths[path][0]])	
					if sensescopy[path][ind] == 'R' and ind != 0:
						collected_buckets.append(bucket_dict[paths[path][0]])
			if deleted:
				goodindex = 0
				badindex = 0
				for i in range(numSamples):
					if labels[i] == 1:
						for c in collected_buckets:
							if c >= numBuckets:
								for j in range(numFeatures):
									A[constraint_cnt,k*numFeatures + j] = I[i][j]
								A[constraint_cnt,zv_ + numBuckets*numBad + (c-numBuckets)*numGood + goodindex] = 1
								rhs.append(1)
								ineq = ineq + "L"
								constraint_cnt = constraint_cnt + 1
								cname = 'GoodAggregateConstraintRight_'+repr(k)+'_Samples_'+repr(i)
								cnames.append(cname)
						goodindex += 1
					else:
						for c in collected_buckets:
							if c < numBuckets:
								for j in range(numFeatures):
									A[constraint_cnt,k*numFeatures + j] = I[i][j]
								A[constraint_cnt,zv_ + c*numBad + badindex] = 1
								rhs.append(1)
								ineq = ineq + "L"
								constraint_cnt = constraint_cnt + 1
								cname = 'BadAggregateConstraintRight_'+repr(k)+'_Samples_'+repr(i)
								cnames.append(cname)
						badindex += 1
			else:
				for i in range(numSamples):
					for c in collected_buckets:
						for j in range(numFeatures):
							A[constraint_cnt,k*numFeatures + j] = I[i][j]
						A[constraint_cnt,zv_ + c*numSamples + i] = 1
						rhs.append(1)				
						ineq = ineq + "L"
						constraint_cnt = constraint_cnt + 1
						cname = 'BucketConstraintRight_'+repr(k)+'_Sample_'+repr(i)+'_Bucket_'+repr(c)
						cnames.append(cname)
	
	print("Number of rows: %s " % constraint_cnt)
	numRows = constraint_cnt
	numCols = zvc_
	print("Number of columns: %s " % numCols)

	indices = [[i for i in range(numRows) if A[i,j] != 0] for j in range(numCols)]
	values = [[A[i,j] for i in range(numRows) if A[i,j] != 0] for j in range(numCols)]
	cols = [[indices[i],values[i]] for i in range(numCols)]

	rhs = np.array(rhs)
	senses = ineq

	# define the objective	
	obj = np.zeros(numCols)
	if deleted:
		for i in range(zv_,zv_+numBuckets*numBad):
			obj[i] = C
		for i in range(zv_+numBuckets*numBad,zvc_):
			obj[i] = 1
	else:	
		for i in range(numSamples):
			if labels[i] == 0:
				for b in range(numBuckets):
					obj[zv_ + b*numSamples + i] = C
			else:
				for b in range(numBuckets):
					obj[zv_ + numBuckets*numSamples + b*numSamples + i] = 1
	
	# set up types, priorities
	priority_vec = []
	
	if not relaxed:	
		types = numNodes*(numGroups+numFeatures)*'I'
		for j in range(numNodes*numFeatures):
			priority_vec.append((j,1,p.order.branch_direction.down))
	else:
		types = ''
		for k in range(numNodes):
			if tree[k].left_child is None and tree[k].right_child is None:
				types = types + numFeatures*'C'
			else:
				types = types + numFeatures*'I'
				for j in range(numFeatures):
					priority_vec.append((k*numFeatures + j,1,p.order.branch_direction.down))
	
		for k in range(numNodes):
			if tree[k].left_child is None and tree[k].right_child is None:
				types = types + numGroups*'I'
			else:
				types = types + numGroups*'I'
			for g in range(numGroups):
				priority_vec.append((z_ + k*numGroups + g, 2 + numNodes - k,p.order.branch_direction.down))
		
	if relaxedobj and deleted:
		types = types + numBuckets*numSamples*'C'
	elif relaxedobj and not deleted:
		types = types + 2*numBuckets*numSamples*'C'
	elif not relaxedobj and deleted:
		types = types + numBuckets*numSamples*'I'
	elif not relaxedobj and not deleted:
		types = types + 2*numBuckets*numSamples*'I'	

	
	lb = np.zeros(numCols)
	ub = np.ones(numCols)
	
	# Load into p
	p.linear_constraints.add(rhs=rhs,senses=senses)
	p.linear_constraints.set_names(zip(range(constraint_cnt),cnames))
	p.variables.add(obj=obj,lb=lb,ub=ub,columns=cols,types=types,names=names)
	if priorities:
		p.order.set(priority_vec)

	p.parameters.timelimit.set(maxtime)
	p.write("dtint.lp")
	p.solve()
	sol = p.solution
	trial = sol.get_objective_value()
	print("Solution value = ",trial)

	if not deleted:
		bucket_counts = np.zeros(2*numBuckets)
		for b in range(2*numBuckets):
			indices = range(zv_ + b*numSamples,zv_ + (b+1)*numSamples)
			bucket_count = 0
			for i in indices:
				if p.solution.get_values(i) == 1:
					bucket_count = bucket_count + 1
			bucket_counts[b] = bucket_count
		for b in range(numBuckets):
			print('Left bucket count ' + repr(b) + ': ' +  repr(bucket_counts[b]))
			print('Right bucket count ' + repr(b) + ': ' + repr(bucket_counts[numBuckets + b]))
	else:
		bucket_counts = np.zeros(2*numBuckets)
		for b in range(numBuckets):
			indices = range(zv_ + b*numBad,zv_ + (b+1)*numBad)
			bucket_count = 0
			for i in indices:
				if p.solution.get_values(i) == 1:
					bucket_count = bucket_count + 1
			bucket_counts[b] = bucket_count
			indices = range(zv_ + numBuckets*numBad + b*numGood,zv_ + numBuckets*numBad + (b+1)*numGood)
			bucket_count = 0
			for i in indices:
				if p.solution.get_values(i) == 1:
					bucket_count = bucket_count + 1
			print('Left bucket count ' + repr(b) + ': ' + repr(bucket_counts[b]))
			bucket_counts[numBuckets + b] = bucket_count
			print('Right bucket count ' + repr(b) + ': ' + repr(bucket_counts[numBuckets+b]))

	# fill up the tree structure:

	# what are the groups in the solution?
	solgroups = []
	splits = [[] for k in range(numNodes)]

	for k in range(numNodes):
		for j in range(z_ + k*numGroups,z_ + (k+1)*numGroups):
			if sol.get_values(j) > 0.99:
				group = j - z_ - k*numGroups
				solgroups.append(group)
	for k in range(numNodes):
		indices = [i for i,x in enumerate(groups) if x == solgroups[k]]
		for ind in indices:
			dir = sol.get_values(k*numFeatures + ind)
			if dir > 0.99:
				splits[k].append(ind)
		tree[k].add_splitvar(splits[k])


def test_int_model(tree,I,labels,gs=False):
	"""
	Given a tree written on by build_int_model and a testset and testlabels, return some statistics
	(recall,specificity,accuracy) about the prediction quality of the model on the testset

	INPUTS:
	tree (required): tree object written on by build_int_model
	I (required): testset
	labels (required): labels corresponding to the testset
	gs (optional): (default = False) indicates whether group structure was used in the construction of tree	

	OUTPUTS:
	returned as a dictionary object, should probably be parsed in a different way future version
	dictionary keys:
	'recall': recall
	'specificity': specificity
	'accuracy': accuracy
	'inds': a numSamples-length binary array indicating which samples in testset are correctly labelled by tree
	"""

	numSamples = len(I)
	numFeatures = len(I[0])

	if not gs:
		# clone
		I2 = np.zeros((numSamples,2*numFeatures))
		for j in range(numFeatures):
			I2[:,2*j] = I[:,j]
			I2[:,2*j+1] = 1-I[:,j]
		numGroups = numFeatures 
		I = I2

	numSamples = len(I)
	type1 = 0
	type2 = 0
	numGood = np.count_nonzero(labels)
	numBad = numSamples - numGood
	correct_inds = [False]*numSamples

	for sample in range(numSamples):
		k = 0
		prevk = 0
		fullstop = 0
		feats = I[sample]
		while fullstop == 0:
			stop = 0
			for lefty in tree[k].splitvar:
				if I[sample][lefty] == 1 and stop == 0:
					prevk = k
					k = tree[k].left_child
					stop = 1
			if stop == 0:
				prevk = k
				k = tree[k].right_child
			if k is None:
				fullstop = 1
		stop = 0
		for lefty in tree[prevk].splitvar:
			if I[sample][lefty] == 1 and stop == 0:
				stop = 1
				if labels[sample] == 0:
					type2 = type2 + 1
					correct_inds[sample] = True
		if stop == 0 and labels[sample] == 1:
			type1 = type1 + 1
			correct_inds[sample]= True
	recall = float(type1)/numGood
	spec = float(type2)/numBad
	acc = float(type1+type2)/numSamples

	return {'recall':recall, 'specificity': spec, 'accuracy': acc, 'inds':correct_inds}

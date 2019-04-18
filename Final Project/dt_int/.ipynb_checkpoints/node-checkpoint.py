class Node:
	def __init__(self,identifier,parent=None,sense=None,splitvar = None,count=0):
		self.__identifier = identifier
		self.__left = None
		self.__right = None

		if parent is not None:
			self.__parent = parent
		else:
			self.__parent = -1

		if splitvar is not None:
			self.__splitvar = splitvar
		
		if sense is not None:
			self.__sense = sense

		if count is not None:
			self.__count = count

	@property
	def identifier(self):
		return self.__identifier

	@property
	def left_child(self):
		return self.__left
	
	@property
	def right_child(self):
		return self.__right
	
	@property
	def splitvar(self):
		return self.__splitvar

	@property
	def parent(self):
		return self.__parent

	@property
	def sense(self):
		return self.__sense

	@property
	def count(self):
		return self.__count

	def add_left_child(self,identifier):
		self.__left = identifier

	def add_right_child(self,identifier):
		self.__right = identifier

	def add_splitvar(self,splitvar):
		self.__splitvar = splitvar

	def add_count(self,count):
		self.__count = count

class Tree:
	def __init__(self):
		self.__nodes = {}
		
	@property
	def nodes(self):
		return self.__nodes

	def add_node(self, identifier, parent = None, sense = None, splitvar = None):
		node = Node(identifier,parent,sense,splitvar)
		self[identifier] = node
		
		if parent is not None:
			if sense == 'L':
				self[parent].add_left_child(identifier)
			elif sense == 'R':
				self[parent].add_right_child(identifier)

		return node
	
	def get_leaf_nodes(self):
		leafs = []
		self._collect_leaf_nodes(self[0],leafs)

		leaves = []
		for leaf in leafs:
			leaves.append(leaf.identifier)
		return leaves

	def _collect_leaf_nodes(self,node,leafs):
		if node is not None:
			if node.left_child == None and node.right_child == None:
				leafs.append(node)
			if node.left_child != None:
				self._collect_leaf_nodes(self[node.left_child],leafs)
			if node.right_child != None:
				self._collect_leaf_nodes(self[node.right_child],leafs)
	
	def __getitem__(self,key):
		return self.__nodes[key]
		
	def __setitem__(self,key,item):
		self.__nodes[key] = item

def depth_k_tree(k):
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

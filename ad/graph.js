var Tensor = require('../tensor.js');
var utils = require('../utils.js');

// Base class for all compute graph nodes
function Node() {
	this.outDegree = 0;
}
Node.prototype.computeOutDegree = function() {};
Node.prototype.backpropImpl = function() {};


// Base class for all nodes with scalar output
function ScalarNode(x) {
	Node.call(this);
	this.x = x;
	this.dx = 0;
}
ScalarNode.prototype = Object.create(Node.prototype);
ScalarNode.prototype.backprop = function() {
	this.dx = 1;
	this.computeOutDegree();
	this.backpropImpl();
};


// Base class for all nodes with tensor output
function TensorNode(x) {
	Node.call(this);
	this.x = x;
	this.dx = new Tensor(x.dims).zero();
}
TensorNode.prototype = Object.create(Node.prototype);
Tensor.prototype.backprop = function() {
	this.dx.fill(1);
	this.computeOutDegree();
	this.backpropImpl();
};


var UnaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing unary functions
	function UnaryNode(x, parent) {
		BaseNode.call(this, x);
		this.parent = parent;
	}
	UnaryNode.prototype = Object.create(BaseNode.prototype);

	UnaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		this.parent.computeOutDegree();
	};

	UnaryNode.prototype.backpropImpl = function() {
		if (--this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			this.parent.backpropImpl();
		}
	}

	return UnaryNode;
});


var BinaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing binary functions
	function BinaryNode(x, parent1, parent2) {
		BaseNode.call(this, x);
		this.parent1 = parent1;
		this.parent2 = parent2;
	}
	BinaryNode.prototype = Object.create(BaseNode.prototype);

	BinaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		this.parent1.computeOutDegree();
		this.parent2.computeOutDegree();
	};

	BinaryNode.prototype.backpropImpl = function() {
		if (--this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			this.parent1.backpropImpl();
			this.parent2.backpropImpl();
		}
	}

	return BinaryNode;
});


var NaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing n-ary functions
	function NaryNode(x, parents) {
		BaseNode.call(this, x);
		this.parents = parents;
	}
	NaryNode.prototype = Object.create(BaseNode.prototype);

	NaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		var n = this.parents.length;
		while (n--) this.parents[n].computeOutDegree();
	};

	NaryNode.prototype.backpropImpl = function() {
		if (--this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			var n = this.parents.length;
			while (n--) this.parents[n].backpropImpl();
		}
	}

	return NaryNode;
});


module.exports = {
	isNode: function(x) { return x instanceof Node; },
	Node: Node,
	ScalarNode: ScalarNode,
	TensorNode: TensorNode,
	UnaryNode: UnaryNode,
	BinaryNode: BinaryNode,
	NaryNode: NaryNode
};



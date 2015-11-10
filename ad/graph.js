var Tensor = require('../tensor.js');
var utils = require('../utils.js');

// Base class for all compute graph nodes
function Node() {}
Node.prototype.computeOutDegree = function() {};
Node.prototype.backpropRec = function() {};
Node.prototype.zeroDerivativesRec = function() {};


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
	this.backpropRec();
};
ScalarNode.prototype.zeroDerivatives = function() {
	this.dx = 0;
	this.zeroDerivativesRec();
};


// Base class for all nodes with tensor output
function TensorNode(x) {
	Node.call(this);
	this.x = x;
	this.dx = new Tensor(x.dims);
}
TensorNode.prototype = Object.create(Node.prototype);
TensorNode.prototype.backprop = function() {
	this.dx.fill(1);
	this.computeOutDegree();
	this.backpropRec();
};
TensorNode.prototype.zeroDerivatives = function() {
	this.dx.zero();
	this.zeroDerivativesRec();
};


var UnaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing unary functions
	function UnaryNode(x, parent) {
		BaseNode.call(this, x);
		this.outDegree = 0;
		this.parent = parent;
	}
	UnaryNode.prototype = Object.create(BaseNode.prototype);

	UnaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		if (this.outDegree === 1) {
			this.parent.computeOutDegree();
		}
	};

	UnaryNode.prototype.backpropRec = function() {
		this.outDegree--;
		if (this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			this.parent.backpropRec();
		}
	};

	UnaryNode.prototype.zeroDerivativesRec = function() {
		this.parent.zeroDerivatives();
	};

	return UnaryNode;
});


var BinaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing binary functions
	function BinaryNode(x, parent1, parent2) {
		BaseNode.call(this, x);
		this.outDegree = 0;
		this.parent1 = parent1;
		this.parent2 = parent2;
	}
	BinaryNode.prototype = Object.create(BaseNode.prototype);

	BinaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		if (this.outDegree === 1) {
			this.parent1.computeOutDegree();
			this.parent2.computeOutDegree();
		}
	};

	BinaryNode.prototype.backpropRec = function() {
		this.outDegree--;
		if (this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			this.parent1.backpropRec();
			this.parent2.backpropRec();
		}
	};

	BinaryNode.prototype.zeroDerivativesRec = function() {
		this.parent1.zeroDerivatives();
		this.parent2.zeroDerivatives();
	};

	return BinaryNode;
});


var NaryNode = utils.memoize(function(BaseNode) {
	// Base class for all nodes representing n-ary functions
	function NaryNode(x, parents) {
		BaseNode.call(this, x);
		this.outDegree = 0;
		this.parents = parents;
	}
	NaryNode.prototype = Object.create(BaseNode.prototype);

	NaryNode.prototype.computeOutDegree = function() {
		this.outDegree++;
		if (this.outDegree === 1) {
			var n = this.parents.length;
			while (n--) this.parents[n].computeOutDegree();
		}
	};

	NaryNode.prototype.backpropRec = function() {
		this.outDegree--;
		if (this.outDegree === 0) {
			this.backward();	// Must be implemented by subclasses
			var n = this.parents.length;
			while (n--) this.parents[n].backpropRec();
		}
	};

	NaryNode.prototype.zeroDerivativesRec = function() {
		var n = this.parents.length;
		while (n--) this.parents[n].zeroDerivatives();
	};

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



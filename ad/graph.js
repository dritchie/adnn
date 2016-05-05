'use strict';

var Tensor = require('../tensor.js');

// Base class for all compute graph nodes
function Node(x, parents, inputs, backward, name) {
	this.x = x;
	this.parents = parents;
	this.inputs = inputs;
	if (backward !== undefined) this.backward = backward;
	this.outDegree = 0;
	this.name = name || 'node';
}
Node.prototype.copy = function(other) {
	this.x = other.x;
	this.parents = other.parents;
	this.inputs = other.inputs;
	if (other.backward !== undefined) this.backward = other.backward;
	this.outDegree = other.outDegree;
	this.name = other.name;
};
Node.prototype.clone = function() {
	var node = Object.create(Node.prototype);
	node.copy(this);
	return node;
};
Node.prototype.computeOutDegree = function() {
	this.outDegree++;
	if (this.outDegree === 1) {
		var n = this.parents.length;
		while (n--) this.parents[n].computeOutDegree();
	}
};
Node.prototype.backpropRec = function() {
	this.outDegree--;
	if (this.outDegree === 0) {
		this.backward();
		var n = this.parents.length;
		while (n--) this.parents[n].backpropRec();
	}
};
Node.prototype.zeroDerivativesRec = function() {
	this.outDegree--;
	if (this.outDegree === 0) {
		this.zeroDerivativesImpl();
		var n = this.parents.length;
		while (n--) this.parents[n].zeroDerivativesRec();
	}
};
Node.prototype.zeroDerivatives = function() {
	this.computeOutDegree();
	this.zeroDerivativesRec();
};
// By default, backward does nothing
Node.prototype.backward = function() {};


// Base class for all nodes with scalar output
function ScalarNode(x, parents, inputs, backward, name) {
	Node.call(this, x, parents, inputs, backward, name || 'scalarNode');
	this.dx = 0;
}
ScalarNode.prototype = Object.create(Node.prototype);
ScalarNode.prototype.copy = function(other) {
	Node.prototype.copy.call(this, other);
	this.dx = other.dx;
};
ScalarNode.prototype.clone = function() {
	var node = Object.create(ScalarNode.prototype);
	node.copy(this);
	return node;
};
ScalarNode.prototype.backprop = function() {
	this.dx = 1;
	this.computeOutDegree();
	this.backpropRec();
};
ScalarNode.prototype.zeroDerivativesImpl = function() {
	this.dx = 0;
};


// Base class for all nodes with tensor output
function TensorNode(x, parents, inputs, backward, name) {
	Node.call(this, x, parents, inputs, backward, name || 'tensorNode');
	this.dx = new Tensor(x.dims);
}
TensorNode.prototype = Object.create(Node.prototype);
TensorNode.prototype.copy = function(other) {
	Node.prototype.copy.call(this, other);
	this.x = other.x.clone();
	this.dx = other.dx.clone();
};
TensorNode.prototype.clone = function() {
	var node = Object.create(Tensor.prototype);
	node.copy(this);
	return node;
};
TensorNode.prototype.refCopy = function(other) {
	Node.prototype.copy.call(this, other);
	this.x = other.x.refClone();
	this.dx = other.dx.refClone();
};
TensorNode.prototype.refClone = function() {
	var node = Object.create(TensorNode.prototype);
	node.refCopy(this);
	return node;
};
TensorNode.prototype.backprop = function() {
	this.dx.fill(1);
	this.computeOutDegree();
	this.backpropRec();
};
TensorNode.prototype.zeroDerivativesImpl = function() {
	this.dx.zero();
};



module.exports = {
	Node: Node,
	ScalarNode: ScalarNode,
	TensorNode: TensorNode
};



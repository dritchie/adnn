var assert = require('assert');
var Network = require('./network.js');
var lift = require('./lift.js');


// A computation involving neural nets can be encapsulated inside a larger
//    neural net. This is essentially function abstraction.
// What follows in this module is basically a little DSL for defining
//    compound functions involving neural nets, by manually constructing the
//    AST for the function. Since our neural nets are (for the time being, at
//    least) just dataflow graphs, the only operator in this language is
//    function composition.
// nn.ast.input() creates an function input AST node.
// Network.compose() creates a function composition AST node.
// nn.ast.compile(inputs, outputs) compiles the AST into a neural network that
//    executes the computation in static single assignment (SSA) form.


function ASTNode(parents) {
	this.parents = parents || [];
}


// When we have a multi-output network (i.e. Array-of-tensors of output)
//    and we want a separate AST node for each output
ASTNode.prototype.split = function(n) {
	var nodes = new Array(n);
	for (var i = 0; i < n; i++) {
		var net = lift(function(tensors) {
			return tensors[i];
		});
		nodes[i] = net.compose(this);
	}
	return nodes;
}


// Adds no new functionality, but convenient to have this be a separate type.
function InputASTNode() {
	ASTNode.call(this);
}
InputASTNode.prototype = Object.create(ASTNode.prototype);

function input() {
	return new InputASTNode();
}


function ComposeASTNode(network, parents) {
	ASTNode.call(this, parents);
	this.network = network;
}
ComposeASTNode.prototype = Object.create(ASTNode.prototype);

Network.prototype.compose = function() {
	return new ComposeASTNode(this, Array.prototype.slice.call(arguments));
};


// ----------------------------------------------------------------------------


function SSANetwork(inputs, outputs) {
	Network.call(this);

	for (var i = 0; i < inputs.length; i++) {
		assert(inputs[i] instanceof InputASTNode,
			'Inputs to composite neural network must be nn.ast.input()');
	}

	// Topological sort
	var visited = [];
	for (var i = 0; i < outputs.length; i++) {
		var fringe = [outputs[i]];
		while (fringe.length > 0) {
			var node = fringe.pop();
			if (!(node instanceof InputASTNode) &&
				visited.indexOf(node) === -1) {
				visited.push(node);
			}
		}
	}
	visited.reverse();

	// Create SSA operations
	this.ops = [];
	this.ins = [];
	this.outs = [];
	for (var i = 0; i < inputs.length; i++) {
		this.ins.push({ output: undefined });
	}
	for (var i = 0; i < visited; i++) {
		var node = visited[i];
		var inputs = node.parents.map(function(p) {
			var i = this.ins.indexOf(p);
			if (i !== -1)
				return this.ins[i];
			i = visited.indexOf(p);
			if (i !== -1)
				return this.ops[i];
			assert(false, 'This should be impossible');
		}.bind(this));
		var op = {
			network: node.network,
			ins: inputs,
			output: undefined
		};
		this.ops.push(op);
		if (outputs.indexOf(node) !== -1) {
			this.outs.push(op);
		}
	}

	// Parameters
	for (var i = 0; i < this.ops.length; i++) {
		this.parameters = this.parameters.concat(this.ops.parameters);
	}
}
SSANetwork.prototype = Object.create(Network.prototype);

SSANetwork.prototype.eval = function() {
	assert(arguments.length === this.ins.length,
		'Incorrect number of arguments to composite neural network');
	for (var i = 0; i < arguments.length; i++) {
		this.ins.output = arguments[i];
	}
	for (var i = 0; i < this.ops.length; i++) {
		var op = this.ops[i];
		var inputs = op.ins.map(function(inop) {
			return inop.output;
		});
		op.output = op.network.eval.apply(null, inputs);
	}
	var outputs = this.outs.map(function(outop) {
		return outop.output;
	});
	return outputs.length === 1 ? outputs[0] : outputs;
};

SSANetwork.prototype.setTraining = function(boolflag) {
	for (var i = 0; i < this.ops.length; i++) {
		this.ops[i].network.setTraining(boolflag);
	}
};


function compile(inputs, ouputs) {
	return new SSANetwork(inputs, ouputs);
}


// ----------------------------------------------------------------------------

// A common composition pattern is a sequence of 1-to-1 networks
function sequence(networks) {
	var inputNode = input();
	var currNode = inputNode;
	for (var i = 0; i < networks.length; i++) {
		currNode = networks[i].compose(currNode);
	}
	return compile([inputNode], [currNode]);
}


module.exports = {
	ast: {
		input: input,
		compile: compile
	},
	sequence: sequence
};



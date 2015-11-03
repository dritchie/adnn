var assert = require('assert');
var Network = require('./network.js');
var lift = require('./lift.js');


// A computation involving neural nets can be encapsulated inside a larger
//    neural net. This is essentially function abstraction.

function compound(fn, subnets, optname) {
	var cnet = new Network();
	cnet.networks = subnets;
	for (var i = 0; i < subnets.length; i++) {
		cnet.parameters = cnet.parameters.concat(subnets[i].parameters);
	}
	cnet.eval = fn;
	cnet.setTraining = function(flag) {
		for (var i = 0; i < this.ops.length; i++) {
			this.networks[i].setTraining(boolflag);
		}
	};
	cnet.name = optname || 'compoundNetwork';
	return cnet;
}


// ----------------------------------------------------------------------------

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
	this.network = network;
	ASTNode.call(this, parents);
}
ComposeASTNode.prototype = Object.create(ASTNode.prototype);

Network.prototype.compose = function() {
	return new ComposeASTNode(this, Array.prototype.slice.call(arguments));
};


function compile(inputs, outputs, optname) {
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
				for (var i = 0; i < node.parents.length; i++) {
					fringe.push(node.parents[i]);
				}
			}
		}
	}
	visited.reverse();

	// Generate SSA code
	var body = 'var args = arguments;\n';
	var outs = [];
	for (var i = 0; i < visited.length; i++) {
		var node = visited[i];
		var ins = node.parents.map(function(p) {
			var i = inputs.indexOf(p);
			if (i !== -1)
				return 'args['+i+']';
			i = visited.indexOf(p);
			if (i !== -1)
				return 'r'+i;
			assert(false, 'impossible');
		});
		body += 'var r'+i+' = this.networks['+i+'].eval('+ins+');\n';
		if (outputs.indexOf(node) !== -1) {
			outs.push('r'+i);
		}
	}
	body += 'return ' + (outs.length > 1 ? '['+outs+']' : outs[0]) + ';';

	// Create compound network
	var fn = new Function(body);
	var networks = visited.map(function(n) { return n.network; });
	optname = optname || 'compiledNetwork';
	return compound(fn, networks, optname);
}


// ----------------------------------------------------------------------------

// A common composition pattern is a sequence of 1-to-1 networks
function sequence(networks, optname) {
	var inputNode = input();
	var currNode = inputNode;
	for (var i = 0; i < networks.length; i++) {
		currNode = networks[i].compose(currNode);
	}
	optname = optname || 'sequenceNetwork';
	return compile([inputNode], [currNode], optname);
}


module.exports = {
	compound: compound,
	ast: {
		input: input,
		compile: compile
	},
	sequence: sequence
};



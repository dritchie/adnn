'use strict';

var assert = require('assert');
var utils = require('../utils.js');
var Network = require('./network.js');


// A computation involving neural nets can be encapsulated inside a larger
//    neural net. This is essentially function abstraction.
// NOTE: Neural nets created this way can only be serialized if 'optname'
//    is provided. Any code attempting to deserialize such a network must
//    first create an instance of one in order to register the deserializer. 
function compound(fn, subnets, optname) {
	function CompoundNetwork(subnets) {
		Network.call(this);
		this.name = optname || 'compoundNetwork';
		this.networks = subnets.slice();
		for (var i = 0; i < subnets.length; i++) {
			this.paramGetters = this.paramGetters.concat(subnets[i].paramGetters);
			this.paramSetters = this.paramSetters.concat(subnets[i].paramSetters);
		}
		// In case this network uses any subnetworks more than once
		this.paramGetters = utils.deduplicate(this.paramGetters);
		this.paramSetters = utils.deduplicate(this.paramSetters);
	}
	CompoundNetwork.prototype = Object.create(Network.prototype);
	CompoundNetwork.prototype.eval = fn;
	CompoundNetwork.prototype.setTraining = function(flag) {
		Network.prototype.setTraining.call(this, flag);
		for (var i = 0; i < this.networks.length; i++) {
			this.networks[i].setTraining(flag);
		}
	};
	if (optname) {
		CompoundNetwork.prototype.serializeJSON = function() {
			return {
				type: optname,
				networks: this.networks.map(function(n) {
					return n.serializeJSON();
				})
			};
		};
		Network.deserializers[optname] = function(json) {
			return new CompoundNetwork(json.networks.map(function(jn) {
				return Network.deserializeJSON(jn);
			}));
		}
	} else {
		CompoundNetwork.prototype.serializeJSON = function() {
			assert(false, 'Cannot serialize unnamed compound network.');
		};
	}
	return new CompoundNetwork(subnets);
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


function ASTNode(type, parents) {
	this.type = type;
	this.parents = parents || [];
}



// When we have a multi-output network (i.e. Array-of-tensors of output)
//    and we want a separate AST node for each outputs
function SelectNetwork(i) {
	Network.call(this);
	this.name = 'select';
	this.i = i;
}
SelectNetwork.prototype = Object.create(Network.prototype);
SelectNetwork.prototype.eval = function(tensors) { return tensors[this.i]; };
SelectNetwork.prototype.serializeJSON = function() {
	return {
		type: 'select',
		i: this.i
	};
};
Network.deserializers.select = function(json) {
	return new SelectNetwork(json.i);
};
ASTNode.prototype.split = function(n) {
	var nodes = new Array(n);
	for (var i = 0; i < n; i++) {
		var net = new SelectNetwork(i);
		nodes[i] = net.compose(this);
	}
	return nodes;
}


function input() {
	return new ASTNode('input');
}


Network.prototype.compose = function() {
	var node = new ASTNode('compose', Array.prototype.slice.call(arguments));
	node.network = this;
	return node;
};


// Actually compiling the AST
function compile(inputs, outputs, optname, optDebug) {
	for (var i = 0; i < inputs.length; i++) {
		assert(inputs[i].type === 'input',
			'Inputs to composite neural network must be nn.ast.input()');
	}

	// Topological sort
	var visited = [];
	for (var i = 0; i < outputs.length; i++) {
		var fringe = [outputs[i]];
		while (fringe.length > 0) {
			var node = fringe.pop();
			if (!(node.type === 'input') && visited.indexOf(node) === -1) {
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
		// Optionally, catch NaNs and Infinities at each intermediate network
		if (optDebug) {
			var ri = 'r'+i;
			var ri_val = ri +'_val';
			var neti = 'this.networks['+i+']';
			body += [
				'var '+ri_val+' = '+ri+'.x || '+ri+';',
				'if (!'+ri_val+'.isFinite().allreduce()) {',
				'	var err = "Non-finite value in output of network '+i+' ("+'+neti+'.name+")\\n";',
				'   err += "Output:\\n";',
				'	err += "["+'+ri_val+'.toString()+"]\\n";',
				'   err += "Inputs:\\n";',
				ins.map(function(input) {
					var inval = '('+input+'.x || ' + input+')';
					return '	err += "["+'+inval+'.toString()+"]\\n";';
				}).join('\n'),
				'   throw new Error(err);',
				'}'
			].join('\n');
		}
		if (outputs.indexOf(node) !== -1) {
			outs.push('r'+i);
		}
	}
	body += 'return ' + (outs.length > 1 ? '['+outs+']' : outs[0]) + ';';

	// Create compound network from compiled function
	var fn = new Function(body);
	var networks = visited.map(function(n) { return n.network; });
	var net = compound(fn, networks);
	net.name = optname || 'compiledNetwork';
	net.nodes = inputs.concat(visited);
	net.inputs = inputs.map(function(n) { return net.nodes.indexOf(n); });
	net.outputs = outputs.map(function(n) { return net.nodes.indexOf(n); });

	net.serializeJSON = function() {
		var jnodes = this.nodes.map(function(n) {
			var jn = {
				type: n.type,
				parents: n.parents.map(function(p) {
					return this.nodes.indexOf(p);
				}.bind(this))
			};
			if (jn.type === 'compose') {
				jn.network = n.network.serializeJSON();
			}
			return jn;
		}.bind(this));
		return {
			type: 'compiled',
			name: this.name,
			nodes: jnodes,
			inputs: this.inputs,
			outputs: this.outputs
		};
	};

	return net;
}
Network.deserializers.compiled = function(json) {
	var nodes = [];
	for (var i = 0; i < json.nodes.length; i++) {
		var jn = json.nodes[i];
		var n = new ASTNode(jn.type, jn.parents.map(function(pi) {
			return nodes[pi];
		}));
		if (jn.type === 'compose') {
			n.network = Network.deserializeJSON(jn.network);
		}
		nodes.push(n);
	}
	var inputs = json.inputs.map(function(i) { return nodes[i]; });
	var outputs = json.outputs.map(function(i) { return nodes[i]; });
	return compile(inputs, outputs, json.name);
};


// ----------------------------------------------------------------------------


// A common composition pattern is a sequence of 1-to-1 networks
function sequence(networks, optname, optDebug) {
	var inputNode = input();
	var currNode = inputNode;
	for (var i = 0; i < networks.length; i++) {
		currNode = networks[i].compose(currNode);
	}
	var net = compile([inputNode], [currNode], undefined, optDebug);
	net.name = optname || 'sequenceNetwork';
	// Could use compile's serialization, but there's a more concise option
	net.serializeJSON = function() {
		return {
			type: 'sequence',
			name: this.name,
			networks: this.networks.map(function(n) {
				return n.serializeJSON();
			})
		};
	}
	return net;
}
Network.deserializers.sequence = function(json) {
	var networks = json.networks.map(function(jn) {
		return Network.deserializeJSON(jn);
	});
	return sequence(networks, json.name);
};


// ----------------------------------------------------------------------------


module.exports = {
	compound: compound,
	ast: {
		input: input,
		compile: compile
	},
	sequence: sequence
};



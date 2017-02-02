'use strict';

var assert = require('assert');
var graph = require('./graph.js');
var Node = graph.Node;
var Tensor = require('../tensor.js');


function checkOutputType(OutputType) {
	assert(OutputType === Tensor || OutputType === Number,
		"Attempting to create AD function with invalid output type '"
		+ OutputType + "'; valid options are 'Number' and 'Tensor'");
}

// Create a new unary AD primitive function
// opts must contain:
//    - OutputType: Number or Tensor
//    - name: name of the operator
//    - forward: Function taking number input, computes number output.
//    - backward: Function taking Node input, computes derivative.
//      Output node available as 'this'.
function newUnaryFunction(opts) {
	var OutputType = opts.OutputType;
	var name = opts.name;
	var forward = opts.forward;
	var backward = opts.backward;
	checkOutputType(OutputType);

	var NodeType = OutputType === Tensor ? graph.TensorNode : graph.ScalarNode;

	function bw() {
		backward.call(this, this.inputs[0]);
	}

	return function(x) {
		if (x instanceof Node) {
			var inputs = [x];
			return new NodeType(forward(x.x), inputs, inputs, bw, name);
		} else {
			return forward(x);
		}
	};
}


// Create a new binary AD primitive function
// opts must contain:
//    - OutputType: Number or Tensor
//    - name: name of the operator
//    - forward: Function taking number inputs, computes number output.
//    - backward1: Function taking (Node, number) inputs, computes derivative
//      of first input. Output node available as 'this'.
//    - backward2: Function taking (number, Node) inputs, computes derivative
//      of second input. Output node available as 'this'.
function newBinaryFunction(opts) {
	var OutputType = opts.OutputType;
	var name = opts.name;
	var forward = opts.forward;
	var backward1 = opts.backward1;
	var backward2 = opts.backward2;
	checkOutputType(OutputType);


	var NodeType = OutputType === Tensor ? graph.TensorNode : graph.ScalarNode;

	function backward11() {
		backward1.call(this, this.inputs[0], this.inputs[1].x);
		backward2.call(this, this.inputs[0].x, this.inputs[1]);
	}
	function backward10() {
		backward1.call(this, this.inputs[0], this.inputs[1]);
	}
	function backward01() {
		backward2.call(this, this.inputs[0], this.inputs[1]);
	}

	return function(x, y) {
		var xIsNode = x instanceof Node;
		var yIsNode = y instanceof Node;
		if (xIsNode && yIsNode) {
			var inputs = [x, y];
			return new NodeType(forward(x.x, y.x), inputs, inputs, backward11, name);
		} else if (xIsNode) {
			return new NodeType(forward(x.x, y), [x], [x, y], backward10, name);
		} else if (yIsNode) {
			return new NodeType(forward(x, y.x), [y], [x, y], backward01, name);
		} else {
			return forward(x, y);
		}
	};
}


// Create a new arbitrary AD primitive function
// opts must contain:
//    - OutputType: Number or Tensor
//    - name: name of the operator
//    - forward: Function taking Node and number inputs, computes number
//      output.
//    - backward: Function taking Node and number inputs, computes
//      derivatives for all Node inputs. Output Node is available as 'this'.
//    - getParents: Function taking Node and number inputs, returns a list
//      of all Node inputs.
function newFunction(opts) {
	var OutputType = opts.OutputType;
	var name = opts.name;
	var forward = opts.forward;
	var backward = opts.backward;
	var getParents = opts.getParents;
	checkOutputType(OutputType);


	var NodeType = OutputType === Tensor ? graph.TensorNode : graph.ScalarNode;

	function bw() {
		backward.apply(this, this.inputs);
	}

	return function() {
		var output = forward.apply(null, arguments);
		var parents = getParents.apply(null, arguments);
		var inputs = Array.prototype.slice.call(arguments);
		var n = parents.length;
		if (n === 0) {
			return output;
		} else {
			return new NodeType(output, parents, inputs, bw, name);
		}
	};
}


// 'getParents' implementation suitable for functions which take an array or
//    a variable number of args, all of which might be Nodes.
function naryGetParents() {
	var args = arguments.length === 1 && arguments[0] instanceof Array ?
		arguments[0] : arguments;
	var p = [];
	var n = args.length;
	while (n--) {
		var arg = args[n];
		if (arg instanceof Node) {
			p.push(arg);
		}
	}
	return p;
}


// Lifting functions which take numbers but don't return numbers to also work
//    on Nodes.
function liftUnaryFunction(f) {
	return function(x) { return f(x instanceof Node ? x.x : x); };
}
function liftBinaryFunction(f) {
	return function(x, y) {
		var xprim = x instanceof Node ? x.x : x;
		var yprim = y instanceof Node ? y.x : y;
		return f(xprim, yprim);
	};
}



module.exports = {
	newUnaryFunction: newUnaryFunction,
	newBinaryFunction: newBinaryFunction,
	newFunction: newFunction,
	naryGetParents: naryGetParents,
	liftUnaryFunction: liftUnaryFunction,
	liftBinaryFunction: liftBinaryFunction
};




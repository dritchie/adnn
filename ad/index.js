'use strict';

var graph = require('./graph.js');
var Tensor = require('../tensor.js');

var Node = graph.Node;
var ScalarNode = graph.ScalarNode;
var TensorNode = graph.TensorNode;

function liftScalar(x, name) { return new ScalarNode(x, [], [], undefined, name); };
function liftTensor(x, name) { return new TensorNode(x, [], [], undefined, name); };
function doLift(x, name) {
	return x instanceof Tensor ? liftTensor(x, name) : liftScalar(x, name);
}

var ad = {
	lift: function(x, name) { return x instanceof Node ? x : doLift(x, name); },
	isLifted: function(x) { return x instanceof Node; },
	value: function(x) { return x instanceof Node ? x.x : x; },
	derivative: function(x) { return x.dx; },
};

// Create randomly-initialized params
ad.params = function(dims, name) {
	return ad.lift(new Tensor(dims).fillRandom(), name); 
};

var func = require('./func.js');
var functions = require('./functions.js');
var modules = [
	func, functions
];
// The macro-transform code only works via node
if (typeof window === "undefined") {
	modules.push(require('./transform.js'));
}
for (var i = 0; i < modules.length; i++) {
	var m = modules[i];
	for (var prop in m) {
		ad[prop] = m[prop];
	}
}

module.exports = ad;
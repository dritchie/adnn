'use strict';


var assert = require('assert');
var utils = require('../utils.js');
var Tensor = require('../tensor.js');


// ----------------------------------------------------------------------------
// Utilities for iterating an arbitrary structure of parameters / gradients


function structType(x) {
	if (x instanceof Tensor) {
		return 'tensor';
	} else if (Array.isArray(x)) {
		return 'array';
	} else if (typeof x === 'object') {
		return 'object';
	} else {
		throw new Error('impossible');
	}
}

function newStructLike(x) {
	var t = structType(x);
	if (t === 'tensor') {
		return new Tensor(x.dims);	// Initializes to zeros
	} else if (t === 'array') {
		return [];
	} else if (t === 'object') {
		return {};
	}
}

// Assumes:
//   - grad has the same structure as param, but it can be missing entries
//   - param has the same structure throughout optimization, but it can
//     gain/lose entries
function foreachGrad(grad, param, otherList, fn) {
	var t = structType(grad);
	if (t === 'tensor') {
		// Base case; apply the function
		var args = [grad, param].concat(otherList);
		fn.apply(args);
	} else if (t === 'array') {
		// Ensure that all arrays in otherList have initialized array items
		for (var i = 0; i < otherList.length; i++) {
			var other = otherList[i];
			var n = grad.length - other.length;
			for (var j = 0; j < n; j++) {
				other.push(newStructLike(grad[j]));
			}
		}
		// Recurse
		for (var i = 0; i < grad.length; i++) {
			foreachGrad(grad[i], param[i], otherList.map(function(a) {
				return a[i];
			}));
		}
	} else if (t === 'object') {
		// Ensure that all objects in otherList have initialized object members
		for (var i = 0; i < otherList.length; i++) {
			var other = otherList[i];
			for (var prop in grad) {
				if (!other.hasOwnProperty(prop)) {
					other[prop] = newStructLike(grad[prop]);
				}
			}
		}
		// Recurse
		for (var prop in grad) {
			foreachGrad(grad[prop], param[prop], otherList.map(function(a) {
				return a[prop];
			}));
		}
	}
}


// ----------------------------------------------------------------------------
// Optimization methods


function sgd(options) {
	options = utils.mergeObjects(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;

	return function(grad, param, step) {
		foreachGrad(grad, param, [], function(g, p) {
			// p = p - stepSize*g;
			p.subeq(g.mul(stepSize));
		});
	};
}

function adagrad(options) {
	options = utils.mergeObjects(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;

	// State
	var g2State;

	return function(grad, param, step) {
		if (g2State === undefined) g2State = newStructLike(grad);
		return foreachGrad(grad, param, [g2State], function(g, p, g2) {
			// g2 = g2 + g*g;
			g2.addeq(g.mul(g));
			// p = p - stepSize*(g / (g2 + 1e-8))
			p.subeq(g.div(g2.sqrt().addeq(1e-8)).muleq(stepSize));
		});
	};
}

// TODO: rmsprop, adam, adadelta


// ----------------------------------------------------------------------------


module.exports = {
	sgd: sgd,
	adagrad: adagrad
};



'use strict';


var assert = require('assert');
var utils = require('../utils.js');
var Tensor = require('../tensor.js');


// Code for handling arbitrary nesting of list/object structures
//    involving Tensors.
// Allows parameters/gradients for optimization to be structured
//    in any way the application needs.


function type(x) {
	if (x instanceof Tensor) {
		return 'tensor';
	} else if (Array.isArray(x)) {
		return 'array';
	} else if (typeof x === 'object') {
		return 'object';
	} else {
		throw new Error('Malformed Tensor structure; expected Tensor, Array, or object, got ' + (typeof x));
	}
}

function map(x, fn) {
	var t = type(x);
	if (t === 'tensor') {
		return fn(x);
	} else if (t === 'array') {
		var ret = [];
		for (var i = 0; i < x.length; i++) {
			ret[i] = map(x[i], fn);
		}
		return ret;
	} else if (t === 'object') {
		var ret = {};
		for (var prop in x) {
			ret[prop] = map(x[prop], fn);
		}
		return ret;
	}
}

function foreach(x, fn) {
	var t = type(x);
	if (t === 'tensor') {
		fn(x);
	} else if (t === 'array') {
		for (var i = 0; i < x.length; i++) {
			foreach(x[i], fn);
		}
	} else if (t === 'object') {
		for (var prop in x) {
			foreach(x[prop], fn);
		}
	}
}

function map2(x, y, fn) {
	var t = type(x);
	// assert(t === type(y));
	if (t === 'tensor') {
		return fn(x, y);
	} else if (t === 'array') {
		// assert(x.length === y.length);
		var ret = [];
		for (var i = 0; i < x.length; i++) {
			ret[i] = map2(x[i], y[i], fn);
		}
		return ret;
	} else if (t === 'object') {
		var ret = {};
		for (var prop in x) {
			// assert(y.hasOwnProperty(prop));
			ret[prop] = map2(x[prop], y[prop], fn);
		}
		return ret;
	}
}

function foreach2(x, fn) {
	var t = type(x);
	// assert(t === type(y));
	if (t === 'tensor') {
		fn(x, y);
	} else if (t === 'array') {
		// assert(x.length === y.length);
		for (var i = 0; i < x.length; i++) {
			foreach2(x[i], y[i], fn);
		}
	} else if (t === 'object') {
		for (var prop in x) {
			// assert(y.hasOwnProperty(prop));
			foreach2(x[prop], y[prop], fn);
		}
	}
}

// Last arg is assumed to be the function
function foreachN() {
	var fn = arguments[arguments.length-1];
	var xs = Array.prototype.slice.call(arguments, 0, arguments.length-1);
	var t = type(xs[0]);
	if (t === 'tensor') {
		fn.apply(null, xs);
	} else if (t === 'array') {
		var n = xs[0].length;
		for (var i = 0; i < n; i++) {
			var args = xs.map(function(x) { return x[i]; });
			args.push(fn);
			foreachN.apply(null, args);
		}
	} else if (t === 'object') {
		for (var prop in xs[0]) {
			var args = xs.map(function(x) { return x[prop]; });
			args.push(fn);
			foreachN.apply(null, args);
		}
	}
}

function zerosLike(x) {
	return map(x, function(xe) {
		return new Tensor([xe.dims]);	// Initializes to zeros
	});
}

function zerosLikeUpdate(tgt, src) {
	var t = type(tgt);
	assert(t === type(src));
	// No need to check Tensor case, because if tgt already has a tensor where
	//    src has one, then we're good.
	if (t === 'array') {
		// Add new zero structs for any array elements tgt is missing
		var n = src.length;
		var m = tgt.length;
		for (var i = m; i < n; i++) {
			tgt.push(zerosLike(src[i]));
		}
		// Recurse into the structs that were already there
		for (var i = 0; i < m; i++) {
			zerosLikeUpdate(tgt[i], src[i]);
		}
	} else if (t === 'object') {
		// Add new zero structs for any object properites tgt is missing,
		//    recurse into the ones that are already there
		for (var prop in src) {
			if (!tgt.hasOwnProperty(prop)) {
				tgt[prop] = zerosLike(src[prop]);
			} else {
				zerosLikeUpdate(tgt[prop], src[prop]);
			}
		}
	}
}

function ensureZerosLike(tgt, src) {
	if (tgt === undefined) {
		return zerosLike(src);
	} else {
		zerosLikeUpdate(tgt, src);
		return tgt;
	}
}



var fns = {};


function addUnaryFunction(fnname) {
	var inplace = new Function('x', [
		'x.'+fnname+'eq();'
	].join('\n'));
	var copying = new Function('x', [
		'return x.'+fnname+'();'
	].join('\n'));

	fns[fnname+'eq'] = function(struct) {
		foreach(struct, inplace);
		return struct;	// For method chaining
	};
	fns[fnname] = function(struct) {
		return map(struct, copying);
	};
}

function addBinaryFunction(fnname) {
	var inplace = new Function('x', 'y', [
		'x.'+fnname+'eq(y);'
	].join('\n'));
	var copying = new Function('x', 'y', [
		'return x.'+fnname+'(y);'
	].join('\n'));

	// When both args are structs
	var inplace_struct_struct = function(struct1, struct2) {
		foreach2(struct1, struct2, inplace);
		return struct1;		// For method chaining
	}
	var copying_struct_struct = function(struct1, struct2) {
		return map2(struct1, struct2, copying);
	}

	// When first arg is a struct but second arg
	//    is a Tensor or a scalar
	var inplace_struct_val = function(struct, val) {
		foreach(struct, function(x) { inplace(x, val); });
		return struct;		// For method chaining
	}
	var copying_struct_val = function(struct, val) {
		return map(struct, function(x) { return copying(x, val); });
	}

	// Register overloaded versions
	fns[fnname+'eq'] = function(x, y) {
		if (typeof y === 'number' || y instanceof Tensor) {
			return inplace_struct_val(x, y);
		} else {
			return inplace_struct_struct(x, y);
		}
	};
	fns[fnname] = function(x, y) {
		if (typeof y === 'number' || y instanceof Tensor) {
			return copying_struct_val(x, y);
		} else {
			return copying_struct_struct(x, y);
		}
	};
}


// We just need these methods for now; we can add more later if needed
addBinaryFunction('add');
addBinaryFunction('div');



module.exports = {
	type: type,
	map: map,
	map2: map2,
	foreach: foreach,
	foreach2: foreach2,
	foreachN: foreachN,
	zerosLike: zerosLike,
	ensureZerosLike: ensureZerosLike
};
module.exports = utils.mergeObjects(module.exports, fns);




'use strict';


var assert = require('assert');
var utils = require('../utils.js');
var Tensor = require('../tensor.js');


// Code for handling arbitrary nesting of list/object structures
//    involving Tensors.
// Allows parameters/gradients for optimization to be structured
//    in any way the application needs.


function type(struct) {
	if (struct instanceof Tensor) {
		return 'tensor';
	} else if (Array.isArray(struct)) {
		return 'array';
	} else if (typeof struct === 'object') {
		return 'object';
	} else {
		throw new Error('Malformed Tensor structure; expected Tensor, Array, or object, got ' + (typeof x));
	}
}

function map(struct, fn) {
	var t = type(struct);
	if (t === 'tensor') {
		return fn(struct);
	} else if (t === 'array') {
		var ret = [];
		for (var i = 0; i < struct.length; i++) {
			ret[i] = map(struct[i], fn);
		}
		return ret;
	} else if (t === 'object') {
		var ret = {};
		for (var prop in struct) {
			ret[prop] = map(struct[prop], fn);
		}
		return ret;
	}
}

// Execute fn for each leaf in struct.
// Iterate through the structs in coIteratees in lockstep and pass their
//    leaves as additional arguments to fn.
// coIteratees is a list of {struct: , ifMissing: } objects, where 'ifMissing'
//    is a function indicating what insert into this coIteratee struct
//    when it is missing a substructure that is present in 'struct'
function foreach(struct, coIteratees, fn) {

	// If there are no coIteratees, then we can use the super simple version
	if (coIteratees.length === 0) {
		function _foreach(struct) {
			var t = type(struct);
			if (t === 'tensor') {
				fn(struct);
			} else if (t === 'array') {
				for (var i = 0; i < struct.length; i++) {
					_foreach(struct[i]);
				}
			} else if (t === 'object') {
				for (var prop in struct) {
					_foreach(struct[prop]);
				}
			}
		}
		_foreach(struct);
	}
	// Otherwise, go with fully-general version
	else {
		var missingFns = coIteratees.map(function(c) { return c.ifMissing; })

		function _foreach(struct, coStructs) {
			var t = type(struct);
			// TODO: assert that all the coIteratees have the same type?
			if (t === 'tensor') {
				fn.apply(null, [struct].concat(coStructs));
			} else if (t === 'array') {
				// Build coStruct lists for recursive calls
				var subCoStructLists = [];
				for (var j = 0; j < struct.length; j++) {
					subCoStructLists.push(coStructs.map(function(s) { return s[j]; }));
				}
				for (var i = 0; i < coStructs.length; i++)
				// Fill in any missing substructs
				for (var i = 0; i < coStructs.length; i++) {
					var coStruct = coStructs[i];
					var missing = missingFns[i];
					var n = struct.length;
					var m = coStruct.length;
					for (var j = m; j < n; j++) {
						coStruct.push(missing(struct[j], subCoStructLists[j]));
					}
				}
				// Recurse
				for (var j = 0; j < struct.length; j++) {
					_foreach(struct[j], subCoStructLists[j]);
				}
			} else if (t === 'object') {
				// Build coStruct lists for recursive calls
				var subCoStructLists = {};
				for (var prop in struct) {
					subCoStructLists[prop] = coStructs.map(function(s) { return s[prop]; });
				}
				// Fill in any missing substructs
				for (var i = 0; i < coStructs.length; i++) {
					var coStruct = coStructs[i];
					var missing = missingFns[i];
					for (var prop in struct) {
						if (!coStruct.hasOwnProperty(prop)) {
							coStruct[prop] = missing(struct[prop], subCoStructLists[prop]);
						}
					}
				}
				// Recurse
				for (var prop in struct) {
					_foreach(struct[prop], subCoStructLists[prop]);
				}
			}
		}

		var coStructs = coIteratees.map(function(c) { return c.struct; });
		_foreach(struct, coStructs, fn);
	}
}

// Possible 'ifMissing' functions
var ifMissing = {
	impossible: function(struct, coStructs) {
		throw new Error('impossible for this struct to have missing elements');
	},
	zeros: function(struct, coStructs) {
		return map(struct, function(x) {
			return new Tensor([x.dims]);	// Initializes to zeros
		});
	},
	copyStruct: function(struct, coStructs) {
		return map(struct, function(x) {
			return x;
		});
	},
	copyCoStruct: function(index) {
		return function(struct, coStructs) {
			return map(coStructs[index], function(x) {
				return x;
			});
		};
	}
}



module.exports = {
	type: type,
	foreach: foreach,
	ifMissing: ifMissing
};




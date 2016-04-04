'use strict';

var Network = require('./network.js');

// Any Tensor-valued AD function can be turned into a neural network with no
//    parameters
function lift(adfn, optname) {
	var net = new Network();
	net.eval = adfn;
	net.name = optname || 'liftedNetwork';
	if (optname) {
		net.serializeJSON = function() {
			return { type: optname };
		};
		Network.deserializers[optname] = function(json) {
			return net;
		};
	} else {
		net.serializeJSON = function() {
			assert(false, 'Cannot serialize unnamed lifted network.');
		}
	}
	return net;
}

// If 'adfn' takes some number of tensor arguments followed by some number of
//    other (parameter-like) arguments, then 'partialeval' returns a Network
//    which is partially-evaluated on those parameter arguments: its 'eval'
//    function takes in just the tensor arguments.
// This is especially useful for turning multi-arg AD functions into
//    parameterized, single-arg networks that can be stacked in sequence.
function partialeval(adfn, name) {
	// Make a partially-evaluated AD function, then lift that
	// Have to provide a custom serializer which records the partially-
	//    evaluated arguments.
	var netCreateFn = function() {
		var partialargs = Array.prototype.slice.call(arguments);
		var fnToLift = function(t) {
			var tensorargs = Array.prototype.slice.call(arguments);
			return adfn.apply(null, tensorargs.concat(partialargs));
		};
		var net = lift(fnToLift);
		net.name = name;
		if (name) {
			net.serializeJSON = function() {
				var json = {
					type: name
				};
				for (var i = 0; i < partialargs.length; i++) {
					json['arg'+i] = partialargs[i];
				}
				return json;
			};
		}
		return net;
	};

	if (name) {
		Network.deserializers[name] = function(json) {
			var args = [];
			for (var prop in json) {
				if (prop.startsWith('arg')) {
					var idx = parseInt(prop.slice(3));
					args[idx] = json[prop];
				}
			}
			return netCreateFn.apply(null, args);
		};
	}

	return netCreateFn;
}

module.exports = {
	lift: lift,
	partialeval: partialeval
};


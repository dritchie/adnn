'use strict';

var assert = require('assert');

// Base class for all neural networks
function Network() {
	this.name = '';
	this.isTraining = false;
	// These arrays store functions which get/set the parameters of the
	//    network (as AD nodes)
	this.paramGetters = [];
	this.paramSetters = [];
}

// Evaluate the function represented by this neural network
Network.prototype.eval = function() {
	assert(false, "Neural networks must implement the 'eval' method.");
};

// Get the parameters of this network
Network.prototype.getParameters = function() {
	// Cache the results
	if (this.__parameters === undefined) {
		this.__parameters = this.paramGetters.map(function(f) {
			return f();
		});
	}
	return this.__parameters;
};

// Set the parameters of this network
Network.prototype.setParameters = function(params) {
	if (params.length !== this.paramSetters.length) {
		assert(false, 'Network.setParameters: size mismatch (network has '
			+ this.paramSetters.length + ' parameters, not ' + params.length);
	}
	for (var i = 0; i < params.length; i++) {
		this.paramSetters[i](params[i]);
	}
	// Replace the cached parameters with the new ones
	this.__parameters = params.slice();
};

// Set whether the network is training or not (this controls whether
//    derivatives will backpropagate through the parameters).
Network.prototype.setTraining = function(boolflag) {
	this.isTraining = boolflag;
};


// Networks can be serialized and deserialized via JSON
Network.prototype.serializeJSON = function() {
	assert(false, "Neural networks must implement the 'serializeJSON' method.");
}
Network.deserializers = {};
Network.deserializeJSON = function(json) {
	assert(json.type !== undefined, "Network JSON blob has no 'type' field.");
	assert(Network.deserializers.hasOwnProperty(json.type),
		"Network JSON blob has unrecognized type '" + json.type + "'");
	return Network.deserializers[json.type](json);
}

module.exports = Network;

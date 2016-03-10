'use strict';

var assert = require('assert');

// Base class for all neural networks
function Network() {
	this.name = '';
	// 'parameters' contains the complete set of parameters for a neural
	//    network, stored as ad.TensorNodes.
	this.parameters = [];
}

// Evaluate the function represented by this neural network
Network.prototype.eval = function() {
	assert(false, "Neural networks must implement the 'eval' method.");
};

// Set whether the network is training or not (this controls whether
//    derivatives will backpropagate through the parameters).
Network.prototype.setTraining = function(boolflag) {};


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

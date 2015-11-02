var assert = require('assert');

// Base class for all neural networks
function Network() {
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


module.exports = Network;

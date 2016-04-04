'use strict';

var Network = require('./network.js');

module.exports = {
	Network: Network,
	deserializeJSON: Network.deserializeJSON
};

// Utilities
var lifting = require('./lifting.js');
var composition = require('./composition.js');

// Networks
var lifted = require('./networks/lifted.js');
var linear = require('./networks/linear.js');
var convolution = require('./networks/convolution.js');
var pooling = require('./networks/pooling.js');
var activation = require('./networks/activation.js');
var perceptron = require('./networks/perceptron.js');
var misc = require('./networks/misc.js');

var modules = [
	lifting, composition, lifted, linear, convolution, pooling, activation, perceptron, misc
];
for (var i = 0; i < modules.length; i++) {
	var m = modules[i];
	for (var prop in m) {
		module.exports[prop] = m[prop];
	}
}
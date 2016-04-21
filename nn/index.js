'use strict';

var utils = require('../utils.js');
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

module.exports = utils.mergeObjects(module.exports,
	lifting, composition, lifted, linear, convolution, pooling, activation, perceptron, misc
);
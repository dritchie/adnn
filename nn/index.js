var Network = require('./network.js');
var lift = require('./lift.js');

module.exports = {
	Network: Network,
	lift: lift
};

// We go ahead and lift all of the Tensor-valued AD functions in
//    ad/functions.js
var adfunctions = require('../ad/functions.js');
for (var fnname in adfunctions.tensor) {
	module.exports[fnname] = lift(adfunctions.tensor[fnname], fnname);
}

// Include everything from composition
var composition = require('./composition.js');
for (var prop in composition) {
	module.exports[prop] = composition[prop];
}

// Other types of networks
var fullyConnected = require('./fullyConnected.js');
module.exports.fullyConnected = fullyConnected;
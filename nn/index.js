var Network = require('./network.js');
var lift = require('./lift.js');
var composition = require('./composition.js');
var adfunctions = require('../ad/functions.js');

module.exports = {
	Network: Network,
	lift: lift
};

// We go ahead and lift all of the Tensor-valued AD functions in
//    ad/functions.js
for (var fnname in adfunctions.tensor) {
	module.exports[fnname] = lift(adfunctions.tensor[fnname], fnname);
}

// Include everything from certain other modules
var modules = [
	composition
];
for (var i = 0; i < modules.length; i++) {
	var m = modules[i];
	for (var prop in m) {
		module.exports[prop] = m[prop];
	}
}
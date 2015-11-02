var Network = require('./network.js');

// Any Tensor-valued AD function can be turned into a neural network with no
//    parameters
function lift(adfn) {
	var net = new Network();
	net.eval = function() { return adfn.apply(null, arguments); };
	return net;
}

module.exports = lift;
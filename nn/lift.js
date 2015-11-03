var Network = require('./network.js');

// Any Tensor-valued AD function can be turned into a neural network with no
//    parameters
function lift(adfn, optname) {
	var net = new Network();
	net.eval = adfn;
	net.name = optname || 'liftedNetwork';
	return net;
}

module.exports = lift;
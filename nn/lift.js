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

module.exports = lift;
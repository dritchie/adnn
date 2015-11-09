var sequence = require('../composition.js').sequence;
var linear = require('./linear.js').linear;


// Convenience function for defining multilayer perceptrons
function mlp(nIn, layerdefs, optname) {
	var nets = [];
	for (var i = 0; i < layerdefs.length; i++) {
		var ldef = layerdefs[i];
		nets.push(linear(nIn, ldef.nOut));
		if (ldef.activation) {
			nets.push(ldef.activation);
		}
		nIn = ldef.nOut;
	}
	return sequence(nets, optname || 'multiLayerPerceptron');
}


module.exports = {
	mlp: mlp
};
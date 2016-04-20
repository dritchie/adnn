'use strict';

var sequence = require('../composition.js').sequence;
var linear = require('./linear.js').linear;


// Convenience function for defining multilayer perceptrons
function mlp(nIn, layerdefs, optname, optDebug) {
	optname = optname || 'mlp';
	var nets = [];
	for (var i = 0; i < layerdefs.length; i++) {
		var ldef = layerdefs[i];
		nets.push(linear(nIn, ldef.nOut, optname+'_layer'+i));
		if (ldef.activation) {
			nets.push(ldef.activation);
		}
		nIn = ldef.nOut;
	}
	return sequence(nets, optname || 'multiLayerPerceptron', optDebug);
}


module.exports = {
	mlp: mlp
};
'use strict';


var utils = require('../utils.js');
var tstruct = require('./tensorStruct.js');



function sgd(options) {
	options = utils.mergeObjects(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;

	return function(grad, param, step) {
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible }
			],
			function(g, p) {
			// p = p - stepSize*g;
			p.subeq(g.mul(stepSize));
		});
	};
}

function adagrad(options) {
	options = utils.mergeObjects(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;

	// State
	var g2State;

	return function(grad, param, step) {
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible },
				{ struct: g2State, ifMissing: tstruct.ifMissing.zeros }
			],
			function(g, p, g2) {
			// g2 = g2 + g*g;
			g2.addeq(g.mul(g));
			// p = p - stepSize*(g / (g2 + 1e-8))
			p.subeq(g.div(g2.sqrt().addeq(1e-8)).muleq(stepSize));
		});
	};
}

// TODO: rmsprop, adam, adadelta



module.exports = {
	sgd: sgd,
	adagrad: adagrad
};



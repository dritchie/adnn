'use strict';


var utils = require('../utils.js');
var tstruct = require('./tensorStruct.js');


var EPS = 1e-8;


function sgd(options) {
	options = utils.mergeDefaults(options, { stepSize: 0.1, stepSizeDecay: 1 });
	var stepSize = options.stepSize;
	var decay = options.stepSizeDecay;

	return function(grad, param, step) {
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible }
			],
			function(g, p) {
				// p = p - stepSize * g;
				p.subeq(g.mul(stepSize));
			}
		);
		stepSize *= decay;
	};
}

function adagrad(options) {
	options = utils.mergeDefaults(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;

	// State
	var g2Struct;

	return function(grad, param, step) {
		if (!g2Struct) g2Struct = tstruct.emptyLike(grad);
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible },
				{ struct: g2Struct, ifMissing: tstruct.ifMissing.zeros },
			],
			function(g, p, g2) {
				// g2 = g2 + g*g;
				g2.addeq(g.mul(g));
				// p = p - stepSize * (g / (sqrt(g2) + 1e-8))
				p.subeq(g.div(g2.sqrt().addeq(EPS)).muleq(stepSize));
			}
		);
	};
}

function rmsprop(options) {
	options = utils.mergeDefaults(options, {stepSize: 0.1, decayRate: 0.9});
    var stepSize = options.stepSize;
    var decayRate = options.decayRate;

    // State
    var g2Struct;

    return function(grad, param, step) {
		if (!g2Struct) g2Struct = tstruct.emptyLike(grad);
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible },
				{ struct: g2Struct, ifMissing: tstruct.ifMissing.zeros },
			],
			function(g, p, g2) {
				// g2 = decayRate*g2 + (1-decayRate)*(g*g)
				g2.muleq(decayRate).addeq(g.mul(g).muleq(1-decayRate));
				// p = p - stepSize * (g / (sqrt(g2) + 1e-8))
				p.subeq(g.div(g2.sqrt().addeq(EPS)).muleq(stepSize));
			}
		);
	};
}

function adam(options) {
	options = utils.mergeDefaults(options, {
    	stepSize: 0.1, // alpha
    	decayRate1: 0.9, // beta1
    	decayRate2: 0.999, // beta2
    });

    var stepSize = options.stepSize;
    var decayRate1 = options.decayRate1;
    var decayRate2 = options.decayRate2;

    var mStruct;
    var vStruct;

    return function(grad, param, step) {
    	var t = step + 1;
    	if (!mStruct) mStruct = tstruct.emptyLike(grad);
    	if (!vStruct) vStruct = tstruct.emptyLike(grad);
    	tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible },
				{ struct: mStruct, ifMissing: tstruct.ifMissing.zeros },
				{ struct: vStruct, ifMissing: tstruct.ifMissing.zeros },
			],
			function(g, p, m, v) {
				// m = decayRate1*m + (1-decayRate1)*g
				m.muleq(decayRate1).addeq(g.mul(1-decayRate1));
				// v = decayRate2*v + (1-decayRate2)*g*g
				v.muleq(decayRate2).addeq(g.mul(g).muleq(1-decayRate2));

				var alpha_t = stepSize * Math.sqrt(1 - Math.pow(decayRate2, t)) / (1 - Math.pow(decayRate1, t));

				// p = p - alpha_t * (m / (sqrt(v) + 1e-8))
				p.subeq(m.div(v.sqrt().addeq(EPS)).muleq(alpha_t));
			}
		);
    }
}



module.exports = {
	sgd: sgd,
	adagrad: adagrad,
	rmsprop: rmsprop,
	adam: adam
};



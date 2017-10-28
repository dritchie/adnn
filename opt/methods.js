'use strict';


var utils = require('../utils.js');
var tstruct = require('./tensorStruct.js');


var EPS = 1e-8;


function sgd(options) {
	options = utils.mergeDefaults(options, { stepSize: 0.1, stepSizeDecay: 1, mu: 0 });
	var stepSize = options.stepSize;
	var stepSizeIsFunction = (typeof stepSize === 'function');
	var decay = options.stepSizeDecay;
	var mu = options.mu; // mu > 0 yields gradient descent with momentum

	// State
	var vStruct;

	return function(grad, param, step) {
		var stepSize_ = stepSizeIsFunction ? stepSize(step) : stepSize;
		if (!vStruct) vStruct = tstruct.emptyLike(grad);
		tstruct.foreach(
			grad,
			[
				{ struct: param, ifMissing: tstruct.ifMissing.impossible },
				{ struct: vStruct, ifMissing: tstruct.ifMissing.zeros },
			],
			function(g, p, v) {
				// v = v * mu - g * stepSize;
				v.muleq(mu).subeq(g.mul(stepSize_));
				// p = p + v
				p.addeq(v);
			}
		);
		if (!stepSizeIsFunction) stepSize *= decay;
	};
}

function adagrad(options) {
	options = utils.mergeDefaults(options, { stepSize: 0.1 });
	var stepSize = options.stepSize;
	var stepSizeIsFunction = (typeof stepSize === 'function');

	// State
	var g2Struct;

	return function(grad, param, step) {
		var stepSize_ = stepSizeIsFunction ? stepSize(step) : stepSize;
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
				p.subeq(g.div(g2.sqrt().addeq(EPS)).muleq(stepSize_));
			}
		);
	};
}

function rmsprop(options) {
	options = utils.mergeDefaults(options, {stepSize: 0.1, decayRate: 0.9});
    var stepSize = options.stepSize;
    var stepSizeIsFunction = (typeof stepSize === 'function');
    var decayRate = options.decayRate;

    // State
    var g2Struct;

    return function(grad, param, step) {
    	var stepSize_ = stepSizeIsFunction ? stepSize(step) : stepSize;
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
				p.subeq(g.div(g2.sqrt().addeq(EPS)).muleq(stepSize_));
			}
		);
	};
}

function adam(options) {
	options = utils.mergeDefaults(options, {
    	stepSize: 0.001, // alpha
    	decayRate1: 0.9, // beta1
    	decayRate2: 0.999, // beta2,
    });

    var stepSize = options.stepSize;
    var stepSizeIsFunction = (typeof stepSize === 'function');
    var decayRate1 = options.decayRate1;
    var decayRate2 = options.decayRate2;

    var mStruct;
    var vStruct;

    return function(grad, param, step) {
    	var stepSize_ = stepSizeIsFunction ? stepSize(step) : stepSize;
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

				var alpha_t = stepSize_ * Math.sqrt(1 - Math.pow(decayRate2, t)) / (1 - Math.pow(decayRate1, t));

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



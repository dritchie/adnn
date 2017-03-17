'use strict';


var assert = require('assert');
var ad = require('../ad');
var utils = require('../utils.js');
var methods = require('./methods.js');
var tstruct = require('./tensorStruct.js');


// fn returns params and grads
function optimize(fn, options) {
	options = utils.mergeDefaults(options, {
		iterations: 100,
		method: methods.sgd(),
		verbose: false
	});

	var iters = options.iterations;
	var method = options.method;
	var verbose = options.verbose;

	for (var i = 0; i < iters; i++) {
		var rets = fn();
		assert(rets.gradients !== undefined);
		assert(rets.parameters !== undefined);
		// TODO: assert that there exists a parameter for each gradient?
		method(rets.gradients, rets.parameters, i);
		if (verbose) {
			console.log('[optimize] done iteration ' + (i+1) + '/' + iters);
		}
	}
}

// Takes an ad function returning { parameters: , loss: }, where both are ad Nodes
// Returns a function which returns { parameters: , gradients: }, where both
//    are Tensors.
// Return function is suitable as input to 'optimize'
function makeOptimizable(fn) {
	return function() {
		var rets = fn();
		assert(rets.parameters !== undefined);
		assert(rets.loss !== undefined);
		// Clear out any derivatives left around from previous iterations
		tstruct.foreach(rets.parameters, [], function(p) {
			p.zeroDerivatives();
		});
		rets.loss.backprop();
		var newrets = {
			gradients: tstruct.map(rets.parameters, ad.derivative),
			parameters: tstruct.map(rets.parameters, ad.value)
		};
		return newrets;
	};
}

// fn is and ad-lifted function which returns { parameters: , loss: }, both
//    of which are ad Nodes
function adOptimize(fn, options) {
	optimize(makeOptimizable(fn), options);
};

// fn is an ad-lifted function which takes some input and returns
//    {output: , parameters: }, both of which are ad Nodes.
// trainingData is list of {input: , output: } tuples
// lossFn is an ad-lifted loss function taking the output of the fn and the
//    ground truth output value and producing a loss
// Constructs minibatches uniformly at random from trainingData
function adTrain(fn, trainingData, lossFn, options) {
	options = utils.mergeDefaults(options, {
		batchSize: 1
	});

	var batchSize = options.batchSize;

	var idx;
	var singleFn = makeOptimizable(function() {
		var trainingDatum = trainingData[idx];
		var rets = fn(trainingDatum.input);
		var loss = lossFn(rets.output, trainingDatum.output);
		return {
			parameters: rets.parameters,
			loss: loss
		};
	});
	var batchFn = function() {
		var gradients;
		var parameters;
		for (var i = 0; i < batchSize; i++) {
			idx = Math.floor(Math.random() * trainingData.length);
			var rets = singleFn();
			if (gradients === undefined) {
				gradients = rets.gradients;
				parameters = rets.parameters;
			} else {
				tstruct.foreach(
					rets.gradients,
					[
						{ struct: gradients, ifMissing: tstruct.ifMissing.zeros },
						{ struct: rets.parameters, ifMissing: tstruct.ifMissing.impossible },
						{ struct: parameters, ifMissing: tstruct.ifMissing.copyFromCoStruct(1) }
					],
					function(newg, g) {
						g.addeq(newg);
					}
				);
			}
		}
		if (batchSize > 1) {
			tstruct.foreach(gradients, [], function(g) {
				g.diveq(batchSize);
			});
		}
		return {
			parameters: parameters,
			gradients: gradients
		};
	}

	optimize(batchFn, options);
}

// Like adTrain, except we use a neural net instead of a lifted ad function
function nnTrain(nn, trainingData, lossFn, options) {
	var fn = function(input) {
		return {
			output: nn.eval(input),
			parameters: nn.getParameters()
		};
	}

	nn.setTraining(true);
	adTrain(fn, trainingData, lossFn, options);
	nn.setTraining(false);
}



module.exports = {
	optimize: optimize,
	adOptimize: adOptimize,
	adTrain: adTrain,
	nnTrain: nnTrain,
};



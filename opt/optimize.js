'use strict';


var assert = require('assert');
var Tensor = require('../tensor.js');
var ad = require('../ad');
var utils = require('../utils.js');
var methods = require('./methods.js');


// fn returns params and grads
function optimize(fn, options) {
	options = utils.mergeObjects(options, {
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
		method(rets.gradients, rets.parameters, i);
		if (verbose) {
			console.log('[optimize] done iteration ' + (i+1) + '/' + iters);
		}
	}
}

// fn returns param Nodes and loss Node
// Creates a function which can be used as input to optimize
function makeOptimizable(fn) {
	return function() {
		var rets = fn();
		assert(rets.parameters !== undefined);
		assert(rets.loss !== undefined);
		rets.loss.backprop();
		return {
			gradients: rets.parameters.map(ad.derivative),
			parameters: rets.parameters.map(ad.value),
		};
	};
}

// trainingData is list of {input: Tensor, output: value} tuples
// lossFn is an ad-lifted loss function taking the output of the nn and the
//    ground truth output value and producing a loss
// Constructs minibatches uniformly at random from trainingData
function nnOptimize(nn, trainingData, lossFn, options) {
	options = utils.mergeObjects(options, {
		batchSize: 1
	});

	var batchSize = options.batchSize;

	var zerosLike = function(tensorList) {
		return tensorList.map(function(t) {
			return new Tensor(t.dims);
		});
	};
	var addEq = function(tensors1, tensors2) {
		for (var i = 0; i < tensors1.length; i++) {
			tensors1[i].addeq(tensors2[i]);
		}
	};
	var divEq = function(tensors, scalar) {
		for (var i = 0; i < tensors.length; i++) {
			tensors[i].diveq(scalar);
		}
	};

	var params = nn.getParameters();
	var paramvals = params.map(ad.value);
	var idx;
	var singleFn = makeOptimizable(function() {
		var trainingDatum = trainingData[idx];
		var output = nn.eval(trainingDatum.input);
		var loss = lossFn(output, trainingDatum.output);
		return {
			parameters: params,
			loss: loss
		};
	});
	var batchFn = function() {
		var gradients = zerosLike(paramvals);
		for (var i = 0; i < batchSize; i++) {
			idx = Math.floor(Math.random() * trainingData.length);
			var rets = singleFn();
			addEq(gradients, rets.gradients);
		}
		divEq(gradients, batchSize);
		return {
			parameters: paramvals,
			gradients: gradients
		};
	}

	nn.setTraining(true);
	optimize(batchFn, options);
	nn.setTraining(false);
}

function classificationLoss(outputProbs, trueClassIndex) {
	var n = ad.value(outputProbs).length;
	assert(trueClassIndex < n,
		'Training datum has true class label ' + trueClassIndex + ', but network only outputs ' + n + ' class probabilities.');
	return ad.scalar.neg(ad.scalar.log(
		ad.tensorEntry(outputProbs, trueClassIndex)));
};

// nn is assumed to have a softmax output
// trainingData 'value' field should be a class index
// Invokes 'nnOptimize' with class negative log-likelihood as the loss function
// TODO: Consider using cross entropy loss instead?
function nnOptimizeClassifier(nn, trainingData, options) {
	optimize(nn, trainingData, classificationLoss, options);
}

function regressionLoss(output, trueOutput) {
	var n = ad.value(output).length;
	var m = trueOutput.length;
	assert(n === m,
		'Network output has different dimensionality than training data (' + n + ' vs. ' + m + ')');
	return ad.tensor.dot(ad.tensor.sub(output, trueOutput));
};

// trainingData 'value' field should be a tensor
// Invokes 'nnOptimize' with squared Euclidean distance as the loss function
function nnOptimizeRegressor(nn, trainingData, options) {
	optimize(nn, trainingData, regressionLoss, options);
}



module.exports = {
	optimize: optimize,
	makeOptimizable: makeOptimizable,
	nnOptimize: nnOptimize,
	classificationLoss: classificationLoss,
	nnOptimizeClassifier: nnOptimizeClassifier,
	regressionLoss: regressionLoss,
	nnOptimizeRegressor: nnOptimizeRegressor
};



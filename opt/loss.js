'use strict';


var assert = require('assert');
var ad = require('../ad');


function classificationLoss(outputProbs, trueClassIndex) {
	var n = ad.value(outputProbs).length;
	assert(trueClassIndex < n,
		'Training datum has true class label ' + trueClassIndex + ', but network only outputs ' + n + ' class probabilities.');
	return ad.scalar.neg(ad.scalar.log(
		ad.tensorEntry(outputProbs, trueClassIndex)));
};

function regressionLoss(output, trueOutput) {
	var n = ad.value(output).length;
	var m = trueOutput.length;
	assert(n === m,
		'Network output has different dimensionality than training data (' + n + ' vs. ' + m + ')');
	var diff = ad.tensor.sub(output, trueOutput);
	return ad.tensor.sumreduce(ad.tensor.mul(diff, diff));
};


module.exports = {
	classificationLoss: classificationLoss,
	regressionLoss: regressionLoss
};
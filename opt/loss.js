'use strict';


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
	return ad.tensor.dot(ad.tensor.sub(output, trueOutput));
};


modules.exports = {
	classificationLoss: classificationLoss,
	regressionLoss: regressionLoss
};
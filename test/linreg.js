var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');
var opt = require('../opt');


// Linear regression test


// Build training data by putting random inputs through a known function
var trueFunc = nn.linear(5, 5);
ad.value(trueFunc.weights).fromArray(
	[[1, 0, 0, 0, 0],
	 [0, 2, 0, 0, 0],
	 [0, 0, 3, 0, 0],
	 [0, 0, 0, 4, 0],
	 [0, 0, 0, 0, 5]]
);
ad.value(trueFunc.biases).fromArray([5, 4, 3, 2, 1]);
var data = [];
var N = 10000;
for (var i = 0; i < N; i++) {
	var x = new Tensor([5]).fillRandom();
	var y = trueFunc.eval(x);
	data.push({
		input: x,
		output: y
	});
}

// Train
var trainFunc = nn.linear(5, 5);
console.time('training');
opt.nnTrain(trainFunc, data, opt.regressionLoss, {
	iterations: 10000,
	batchSize: 1,
	method: opt.sgd({ stepSize: 1, stepSizeDecay: 0.999 }),
	// method: opt.adagrad({ stepSize: 1 }),
	// method: opt.rmsprop(),
	// method: opt.adam(),
	verbose: false
});
console.timeEnd('training');
console.log(ad.value(trainFunc.weights).toArray());
console.log(ad.value(trainFunc.biases).toArray());



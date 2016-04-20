var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');


// Linear regression test


// Build training data by putting random inputs
// through a known function
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
trainFunc.setTraining(true);
function dist(x1, x2) {
	var diff = ad.tensor.sub(x1, x2);
	var mul = ad.tensor.mul(diff, diff);
	var mulcomps = ad.tensorToScalars(mul);
	var dot = ad.scalar.sum(mulcomps);
	return ad.scalar.sqrt(dot);
}
var learnRate = 1;
for (var iter = 0; iter < 20; iter++) {
	for (var i = 0; i < N; i++) {
		var datum = data[i];
		var y = trainFunc.eval(datum.input);
		var loss = dist(y, datum.output);
		loss.backprop();
		var prms = trainFunc.getParameters().map(ad.value);
		var grad = trainFunc.getParameters().map(ad.derivative);
		for (var k = 0; k < prms.length; k++) {
			var prm = trainFunc.getParameters()[k];
			var p = ad.value(prm);
			var g = ad.derivative(prm);
			g.muleq(learnRate);
			p.subeq(g);
			prm.zeroDerivatives();
		}
	}
	learnRate *= 0.9;
	console.log('------------------------');
	console.log(ad.value(trainFunc.weights).toArray());
	console.log(ad.value(trainFunc.biases).toArray());
}




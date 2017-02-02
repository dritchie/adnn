var Tensor = require('../newtensor');
//var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');
var opt = require('../opt');

// Tensor multiplication 

var t = new Tensor([2,2]);
t.fromArray([[1,2],[4,5]]);
var t_1 = new Tensor([2,2]);
t_1.fromArray([[0,3],[4,8]])
//console.log(t_1)
//var z = new Tensor([2,2]).zero;
//return
//var t_1 = new Tensor([2]);
//var dat = t_1.fromArray([10,13]);

var N = 2;
//var mat = new Tensor([N,N]).fillRandom();
var mat = new Tensor([N,N]).fill(3.0)
console.log(mat.toArray());
//console.log("filled", mat.fill(3.0).data);
//console.log(mat.ls_to_array(mat.size));
console.log(mat.toFlatArray());
console.log(mat.sum());
//var mat_1 = new Tensor([N,N]).fillRandom();
var mat_1 = new Tensor([N,N]).fill(2);
console.log('starting dot')
console.time('tensor');
//var res = data.min();
//remap  :r !pbpaste
//
//
//var res = mat.mul(mat_1);
var res = mat.dot(mat_1);
//console.log(res.toFlatArray());

// Train
//var trainFunc = nn.linear(5, 5);
//console.time('tensor');
//opt.nnTrain(trainFunc, data, opt.regressionLoss, {
//	iterations: 10000,
//	batchSize: 1,
//	method: opt.sgd({ stepSize: 1, stepSizeDecay: 0.999 }),
//	// method: opt.adagrad({ stepSize: 1 }),
//	// method: opt.rmsprop(),
//	// method: opt.adam(),
//	verbose: false
//});
console.timeEnd('tensor');

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

var N = 1000;
var mat = new Tensor([N,N]).fillRandom();
var mat_1 = new Tensor([N,N]).fillRandom();
console.time('tensor');
//var res = data.min();
//var res = mat.mul(mat_1);
//console.log("testbefore", mat.toFlatArray());
//res = mat.fill(12);
//console.log("testafter", mat.toFlatArray());
var res = mat.dot(mat_1);
//console.log(res.toArray());

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

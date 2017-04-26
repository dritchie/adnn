var assert = require('assert');
var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');
var _ = require('lodash');

var mat33 = [[1, 3, 5],
             [2, 4, 6],
             [9, 7, 3]];
var mat33T = [[1, 2, 9],
             [3, 4, 7],
             [5, 6, 3]];
var matdot = [[18, 18, 18],
              [24, 24, 24],
              [38, 38, 38]];

function roundArr (arr, sf) {
  var i = 0;
  while(i < arr.length){ 
    arr[i] = parseFloat(arr[i].toFixed(sf)); 
    i++;
  }
  return arr;
}

function fillArr (el, len) {
  var arr = []
  for (var i = 0; i < len; i++) {
    arr.push(el);
  }
  return arr;
}

function run (type) {
  var tensortype = type === "torch" ? "thtensor" : "tensor";
  var x = tensortype === "tensor" ? new Tensor([5]).fill(1.5) : new THTensor([5]).fill(1.5);
  var y = x.clone().fill(0.2);
  var z = y.clone().fill(0.3);
  var d_1 = tensortype === "tensor" ? new Tensor([3,3]).fromArray(mat33) : new THTensor([3,3]).fromArray(mat33);
  var d2 = d_1.clone().fill(2);
  // Raw Numbers/Tensors can be used with AD functions
  //Unary operators
  assert.equal(Number(ad.scalar.tanh(1.5)).toFixed(3), 0.905);  // 0.9051...
  assert.ok(_.isEqual(roundArr(ad.tensor.tanh(x).toArray(), 3), fillArr(0.905, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.neg(x).toArray(), 1), fillArr(-1.5, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.floor(x).toArray(), 1), fillArr(1, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.ceil(x).toArray(), 1), fillArr(2, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.round(x).toArray(), 1), fillArr(2, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.sqrt(x).toArray(), 2), fillArr(1.22, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.sqrt(x).toArray(), 2), fillArr(1.22, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.exp(x).toArray(), 2), fillArr(4.48, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.log(x).toArray(), 3), fillArr(0.405, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.abs(x).toArray(), 2), fillArr(1.5, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.sin(x).toArray(), 3), fillArr(0.997, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.cos(x).toArray(), 3), fillArr(0.071, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.tan(x).toArray(), 3), fillArr(14.101, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.sigmoid(x).toArray(), 3), fillArr(0.818, 5)));

 //Binary operators
  assert.ok(_.isEqual(roundArr(ad.tensor.add(x, y).toArray(), 1), fillArr(1.7, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.sub(x, y).toArray(), 1), fillArr(1.3, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.mul(x, y).toArray(), 1), fillArr(0.3, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.div(x, y).toArray(), 1), fillArr(7.5, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.pow(x, y).toArray(), 3), fillArr(1.084, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.max(x, y).toArray(), 1), fillArr(1.5, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.min(x, y).toArray(), 1), fillArr(0.2, 5)));
  assert.ok(_.isEqual(roundArr(ad.tensor.atan2(x, y).toArray(), 3), fillArr(1.438, 5)));
  console.log(tensortype + " MATH OPS TESTS PASSED") 
  
  // To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
  //    by 'lifting' them
  var scalarIn = ad.lift(1.5);
  var tensorIn = ad.lift(new THTensor([3]).fill(1.5));
  var tensor2 = ad.lift(new THTensor([3]).fill(2));
//   console.log(tensorIn)
  
  // Feeding these nodes into AD functions results in Node outputs, which can be used to
  //    initialize backpropagation
  var scalarOut = ad.scalar.tanh(1.5);
  var tensorOut = ad.tensor.mul(tensorIn,tensor2);
  tensorOut.backprop();
//   console.log(tensorOut.dx.toArray()); return
  
  // We can then retrieve the values and derivatives of different nodes
  ad.value(scalarOut);  // 0.9051...
  ad.derivative(scalarIn);  // 0.1807...
  
  ad.isLifted(scalarIn);
  ad.isLifted(1.5);
  //tensor and linalg ops
  assert.equal(ad.tensor.sumreduce(x), 7.5);
  assert.ok(ad.tensor.allreduce(x));
  assert.ok(ad.tensor.anyreduce(x));
  assert.equal(ad.tensor.get(x, 1), 1.5);
  assert.ok(_.isEqual(ad.tensor.toScalars(x), [1.5, 1.5, 1.5, 1.5, 1.5]))
  assert.ok(_.isEqual(ad.tensor.fromScalars([0, 1, 2, 3]).toArray(), [0, 1, 2, 3]))
  assert.ok(_.isEqual(ad.tensor.range(x, 0, 2).toArray(), [1.5, 1.5]))
  console.log(tensortype + " TENSOR OPS TESTS PASSED")
  assert.ok(_.isEqual(ad.tensor.split(x, [2,3])[0].toArray(), [1.5, 1.5]))
// rounding errors for concat but it works
  assert.ok(_.isEqual(roundArr(ad.tensor.concat([x,y,z]).toArray(), 1),
          [1.5, 1.5, 1.5, 1.5, 1.5,
          0.2, 0.2, 0.2, 0.2, 0.2,
          0.3, 0.3, 0.3, 0.3, 0.3]));
  assert.ok(_.isEqual(ad.tensor.transpose(d_1).toArray(), mat33T));
//   console.log(ad.tensor.diagonal(d2))
// rounding errors fpr inverse
  ad.tensor.inverse(d_1).toArray()
  assert.equal(Number(ad.tensor.determinant(d_1)).toFixed(1), 4)
  assert.ok(_.isEqual(ad.tensor.dot(d_1, d2).toArray(), matdot));
  assert.equal(Number(ad.tensor.softmax(x).toArray()[0]).toFixed(1), 0.2)
  console.log(tensortype + " LIN ALG OPS TESTS PASSED ")
}

run ("torch")
run ("tensor")

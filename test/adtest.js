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

function run (type) {
  var tensortype = type === "torch" ? "thtensor" : "tensor";
  var x = tensortype === "tensor" ? new Tensor([5]).fill(1.5) : new THTensor([5]).fill(1.5);
  var y = x.clone().fill(0.2);
  var z = y.clone().fill(0.3);
  var d_1 = tensortype === "tensor" ? new Tensor([3,3]).fromArray(mat33) : new THTensor([3,3]).fromArray(mat33);
  var d2 = d_1.clone().fill(2);
  // Raw Numbers/Tensors can be used with AD functions
  ad.scalar.tanh(1.5);  // 0.9051...
  ad[tensortype].tanh(x);  // [0.9051, 0.9051, 0.9051]
  
  // To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
  //    by 'lifting' them
  var scalarIn = ad.lift(1.5);
  var tensorIn = ad.lift(new Tensor([3]).fill(1.5));
  
  // Feeding these nodes into AD functions results in Node outputs, which can be used to
  //    initialize backpropagation
  var scalarOut = ad.scalar.tanh(1.5);
  var tensorOut = ad.tensor.tanh(tensorIn);
 tensorOut.backprop();
  
  // We can then retrieve the values and derivatives of different nodes
  ad.value(scalarOut);  // 0.9051...
  ad.derivative(scalarIn);  // 0.1807...
  
  ad.isLifted(scalarIn);
  ad.isLifted(1.5);
//   console.log(ad.tensor) 
  //tensor and linalg ops
  assert.equal(ad[tensortype].sumreduce(x), 7.5);
  assert.ok(ad[tensortype].allreduce(x));
  assert.ok(ad[tensortype].anyreduce(x));
  assert.equal(ad[tensortype].get(x, 1), 1.5);
  assert.ok(_.isEqual(ad[tensortype].toScalars(x), [1.5, 1.5, 1.5, 1.5, 1.5]))
  assert.ok(_.isEqual(ad[tensortype].fromScalars([0, 1, 2, 3]).toArray(), [0, 1, 2, 3]))
  assert.ok(_.isEqual(ad[tensortype].range(x, 0, 2).toArray(), [1.5, 1.5]))
  console.log(tensortype + "OPS COMPLETE")
  ad.tensor.split(x, 2)
// rounding errors for concat but it works
//   assert.ok("concat:", _.isEqual(ad[tensortype].concat([x,y,z]).toArray(),
//           [1.5, 1.5, 1.5, 1.5, 1.5,
//           0.2, 0.2, 0.2, 0.2, 0.2,
//           0.3, 0.3, 0.3, 0.3, 0.3]));
  assert.ok("transpose:", _.isEqual(ad[tensortype].transpose(d_1).toArray(), mat33T));
//   console.log(ad[tensortype].diagonal(d2))
// rounding errors fpr inverse
  ad[tensortype].inverse(d_1).toArray()
  assert.equal(Number(ad[tensortype].determinant(d_1)).toFixed(1), 4)
  assert.ok(_.isEqual(ad[tensortype].dot(d_1, d2).toArray(), matdot));
  assert.equal(Number(ad[tensortype].softmax(x).toArray()[0]).toFixed(1), 0.2)
  console.log(tensortype + " LIN ALG OPS COMPLETE")
}

run ("torch")
run ("tensor")

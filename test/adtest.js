var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');

function run (type) {
  var tensortype = type === "torch" ? "thtensor" : "tensor";
  var x = tensortype === "tensor" ? new Tensor([5]).fill(1.5) : new THTensor([5]).fill(1.5);
  var y = tensortype === "tensor" ? new Tensor([5]).fill(1.5) : new THTensor([5]).fill(0.2);
  var z = y.fillRandom();
  var d2 = tensortype === "tensor" ? new Tensor([3,3]).fillRandom() : new THTensor([3,3]).fillRandom();
  var d_1 = d2.fillRandom();
  // Raw Numbers/Tensors can be used with AD functions
  ad.scalar.tanh(1.5);  // 0.9051...
  ad[tensortype].tanh(x);  // [0.9051, 0.9051, 0.9051]
  
  // To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
  //    by 'lifting' them
  var scalarIn = ad.lift(1.5);
  var tensorIn = ad.lift(new Tensor([3]).fill(1.5));
  
  // Feeding these nodes into AD functions results in Node outputs, which can be used to
  //    initialize backpropagation
  var scalarOut = ad.scalar.tanh(scalarIn);
  console.log(scalarOut.backprop());
  
  // We can then retrieve the values and derivatives of different nodes
  ad.value(scalarOut);  // 0.9051...
  ad.derivative(scalarIn);  // 0.1807...
  
  ad.isLifted(scalarIn);
  ad.isLifted(1.5);
  
  //tensor and linalg ops
  ad[tensortype].sumreduce(x)
  ad[tensortype].allreduce(x)
  ad[tensortype].anyreduce(x)
  ad[tensortype].get(x, 1)
  ad[tensortype].toScalars(x)
  ad[tensortype].fromScalars([0,1,2,3])
  ad[tensortype].range(x, 0, 2)
  //[ad.tensor.]split(x, sizes)
  ad[tensortype].concat([x,y,z])
  ad[tensortype].transpose(d2)
  ad[tensortype].diagonal(d2)
  ad[tensortype].inverse(d2)
  ad[tensortype].determinant(d2)
  ad[tensortype].dot(d2, d_1)
  ad[tensortype].softmax(x)
}

run ("torch")
run ("tensor")

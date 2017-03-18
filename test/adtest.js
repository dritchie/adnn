var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');

// var x = new Tensor([5]).fill(1.5)
// var y = new Tensor([3]).fill(0.2)
// var z = y.fillRandom()
// // Raw Numbers/Tensors can be used with AD functions
// ad.scalar.tanh(1.5);  // 0.9051...
// ad.tensor.tanh(x);  // [0.9051, 0.9051, 0.9051]
// 
// // To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
// //    by 'lifting' them
// var scalarIn = ad.lift(1.5);
// var tensorIn = ad.lift(new Tensor([3]).fill(1.5));
// 
// // Feeding these nodes into AD functions results in Node outputs, which can be used to
// //    initialize backpropagation
// var scalarOut = ad.scalar.tanh(scalarIn);
// scalarOut.backprop();
// 
// // We can then retrieve the values and derivatives of different nodes
// ad.value(scalarOut);  // 0.9051...
// ad.derivative(scalarIn);  // 0.1807...
// 
// ad.isLifted(scalarIn);
// ad.isLifted(1.5);
// 
// //tensor and linalg ops
// ad.tensor.sumreduce(x)
// ad.tensor.allreduce(x)
// ad.tensor.anyreduce(x)
// ad.tensor.get(x, 1)
// ad.tensor.toScalars(x)
// ad.tensor.fromScalars([0,1,2,3])
// ad.tensor.range(x, 0, 2)
// // ad.tensor.split(x, sizes)
// ad.tensor.concat([x,y,z])
// // ad.tensor.transpose(x)
// ad.tensor.diagonal(x)
// ad.tensor.inverse(x)
// ad.tensor.determinant(x)
// ad.tensor.dot(x, y)
// ad.tensor.softmax(x)

// TORCH TEST
console.log("torch AD tests")
var x = new THTensor([5]).fill(1.5)
var y = new THTensor([3]).fill(0.2)
var z = y.fillRandom()
var d2 = new THTensor([3,3]).fillRandom()
var d_1 = new THTensor([3,3]).fillRandom()
// Raw Numbers/Tensors can be used with AD functions
ad.scalar.tanh(1.5);  // 0.9051...
ad.thtensor.tanh(x);  // [0.9051, 0.9051, 0.9051]

// To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
//    by 'lifting' them
var scalarIn = ad.lift(1.5);
var tensorIn = ad.lift(new Tensor([3]).fill(1.5));

// Feeding these nodes into AD functions results in Node outputs, which can be used to
//    initialize backpropagation
var scalarOut = ad.scalar.tanh(scalarIn);
scalarOut.backprop();

// We can then retrieve the values and derivatives of different nodes
ad.value(scalarOut);  // 0.9051...
ad.derivative(scalarIn);  // 0.1807...

ad.isLifted(scalarIn);
ad.isLifted(1.5);

//tensor and linalg ops
ad.thtensor.sumreduce(x)
ad.thtensor.allreduce(x)
ad.thtensor.anyreduce(x)
ad.thtensor.get(x, 1)
ad.thtensor.toScalars(x)
ad.thtensor.fromScalars([0,1,2,3])
ad.thtensor.range(x, 0, 2)
// ad.tensor.split(x, sizes)
ad.thtensor.concat([x,y,z])
ad.thtensor.transpose(d2)
ad.thtensor.diagonal(d2)
ad.thtensor.inverse(d2)
ad.thtensor.determinant(d2)
ad.thtensor.dot(d2, d_1)
ad.thtensor.softmax(x)

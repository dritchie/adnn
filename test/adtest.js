var ad = require('../ad');

var x = new THTensor([5]).fill(1.5)
// Raw Numbers/Tensors can be used with AD functions
ad.scalar.tanh(1.5);  // 0.9051...
ad.tensor.tanh(x);  // [0.9051, 0.9051, 0.9051]

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

ad.tensor.sumreduce(x)
ad.tensor.allreduce(x)
ad.tensor.anyreduce(x)
ad.tensor.get(x, 1)
ad.tensor.toScalars(x)
ad.tensor.fromScalars([0,1,2,3])
ad.tensor.range(x, i, j)
ad.tensor.split(x, sizes)
ad.tensor.concat(lst)
ad.tensor.transpose(x)
ad.tensor.diagonal(x)
ad.tensor.inverse(x)
ad.tensor.determinant(x)
//ad.tensor.dot(x, y)
ad.tensor.softmax(x)

# adnn
adnn provides Javascript-native neural networks on top of general scalar/tensor reverse-mode automatic differentiation. You can use just the AD code, or the NN layer built on top of it. This architecture makes it easy to define big, complex numerical computations and compute derivatives w.r.t. their inputs/parameters.

adnn is not an optimization library--it does not (currently) provide any methods for optimizing the parameters of neural nets / big differentiable computations (e.g. SGD, AdaGrad). However, these algorithms are simple to implement on top of adnn.

### Examples ###

#### Scalar code ####

The simplest use case for adnn:

````javascript
var ad = require('adnn/ad')

function dist(x1, y1, x2, y2) {
  var xdiff = ad.scalar.sub(x1, x2);
  var ydiff = ad.scalar.sub(y1, y2);
  return ad.scalar.sqrt(ad.scalar.add(
    ad.scalar.mul(xdiff, xdiff),
    ad.scalar.mul(ydiff, ydiff)
  ));
}

// Can use normal scalar inputs
var out = dist(0, 1, 1, 4);
console.log(out);   // 3.162...

// Use 'lifted' inputs to track derivatives
var x1 = ad.lift(0);
var y1 = ad.lift(1);
var x2 = ad.lift(1);
var y2 = ad.lift(4);
var out = dist(x1, y1, x2, y2);
console.log(ad.value(out));   // still 3.162...
out.backprop();   // Compute derivatives of inputs
console.log(ad.derivative(x1)); // -0.316...
````

It is also possible to write normal Javascript code that uses math operators such as `+` and `*`; adnn can transform this code using a [Sweet.js](http://sweetjs.org/) macro:

````javascript
// In a file called 'dist.js':
function dist(x1, y1, x2, y2) {
  var xdiff = x1 - x2;
  var ydiff = y1 - y2;
  return Math.sqrt(xdiff*xdiff + ydiff*ydiff);
}
module.exports = dist;

// -------------------------------------

// In a separate file:
var ad = require('adnn/ad');
var dist = ad.macroRequire('./dist.js');

var x1 = ad.lift(0);
var y1 = ad.lift(1);
var x2 = ad.lift(1);
var y2 = ad.lift(4);
var out = dist(x1, y1, x2, y2);
console.log(ad.value(out));   // 3.162...
out.backprop();
console.log(ad.derivative(x1)); // -0.316...
````

#### Tensor code ####

adnn also supports computations involving tensors, or a mixture of scalars and tensors:

````javascript
var ad = require('adnn/ad');
var Tensor = require('adnn/tensor');

function dot(vec) {
  var sq = ad.tensor.mul(vec, vec);
  return ad.scalar.sum(ad.tensorToScalars(sq));
}

function dist(vec1, vec2) {
  return ad.scalar.sqrt(dot(ad.tensor.sub(vec1, vec2)));
}

var vec1 = ad.lift(new Tensor([3]).fromFlatArray([0, 1, 1]));
var vec2 = ad.lift(new Tensor([3]).fromFlatArray([2, 0, 3]));
var out = dist(vec1, vec2);
console.log(ad.value(out));   // 3
out.backprop();
console.log(ad.derivative(vec1).toFlatArray());  // [-0.66, 0.33, -0.66]
````

#### Simple neural network ####

adnn makes it easy to define simple, feedforward neural networks. Here's a basic multilayer perceptron that takes a feature vector as input and outputs class probabilities:

````javascript
var Tensor = require('adnn/Tensor');
var ad = require('adnn/ad');
var nn = require('adnn/nn');

// A simple softmax layer
var softmax = nn.lift(function(vec) {
});

var nInputs = 20;
var nHidden = 10;
var nClasses = 5;

// Simple definition using 'nn.mlp' utility
var mlpnet = nn.mlp(nInputs, [
  {nOut: nHidden, activation: nn.tanh},
  {nOut: nClasses}
]);

````

#### Convolutional neural network ####

#### Recurrent neural network ####

### The `ad` module ###

### The `nn` module ###

### Tensors ###

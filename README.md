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
  return ad.sumreduce(sq);
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

var nInputs = 20;
var nHidden = 10;
var nClasses = 5;

// Definition using basic layers
var net = nn.sequence([
  nn.linear(nInputs, nHidden),
  nn.tanh,
  nn.linear(nHidden, nClasses),
  nn.softmax
]);

// Alternate definition using 'nn.mlp' utility
net = nn.sequence([
  nn.mlp(nInputs, [
    {nOut: nHidden, activation: nn.tanh},
    {nOut: nClasses}
  ]),
  nn.softmax
]);

// Enable training
net.setTraining(true);

// Evaluate the network on some features
var features = ad.lift(new Tensor([nInputs]).fillRandom());
var classProbs = net.eval(features);
// Compute gradient w.r.t. log probability of the true class
var trueClass = 3;
var trueClassLP = ad.scalar.log(ad.tensorEntry(classProbs, trueClass));
trueClassLP.backprop();
// Access parameter gradients
var gradients = net.parameters.map(function(pvec) { return ad.derivative(pvec); };

````

#### Convolutional neural network ####

adnn includes the buildings blocks necessary to create convolutional networks. Here is a simple example, adapted from a [ConvNetJS](https://github.com/karpathy/convnetjs) example:

````javascript
var nn = require('adnn/nn');

var net = nn.sequence([
  // Assumes inputs are 32x32 RGB images (i.e. 3x32x32 Tensors)
  nn.convolution({inDepth: 3, outDepth: 16, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 16x16x16
  nn.convolution({inDepth: 16, outDepth: 20, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 20x8x8
  nn.convolution({inDepth: 20, outDepth: 20, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 20x4x4 = 320
  nn.linear(320, 10),
  nn.softmax
  // Output is 10 class probabilities
]);

````

#### Recurrent neural network ####

adnn is also flexible enough to support recurrent neural networks. Here's an example of a rudimentary RNN:

````javascript
var ad = require('adnn/ad');
var nn = require('adnn/nn');

var inputSize = 10;
var outputSize = 5;
var stateSize = 20;

// Component neural networks used by the RNN
var inputNet = nn.linear(inputSize, stateSize);
var stateNet = nn.linear(stateSize, stateSize);
var outputNet = nn.linear(stateSize, outputSize);
var initialStateNet = nn.constantparams([stateSize]);

function processSequence(seq) {
  // Initialize hidden state
  var state = initialStateNet.eval();
  // Process input sequence in order
  var outputs = [];
  for (var i = 0; i < seq.length; i++) {
    // Update hidden state
    state = ad.tensor.tanh(ad.tensor.add(inputNet.eval(seq[i]), stateNet.eval(state)))
    // Generate output
    outputs.push(outputNet.eval(state));
  }
}
````

### The `ad` module ###

### The `nn` module ###

### Tensors ###

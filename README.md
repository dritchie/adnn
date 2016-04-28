# adnn
adnn provides Javascript-native neural networks on top of general scalar/tensor reverse-mode automatic differentiation. You can use just the AD code, or the NN layer built on top of it. This architecture makes it easy to define big, complex numerical computations and compute derivatives w.r.t. their inputs/parameters. adnn also includes utilities for optimizing/training the parameters of such computations.

### Examples

#### Scalar code

The simplest use case for adnn:

```javascript
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
```

It is also possible to write normal Javascript code that uses math operators such as `+` and `*`; adnn can transform this code using a [Sweet.js](http://sweetjs.org/) macro:

```javascript
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
```

#### Tensor code

adnn also supports computations involving tensors, or a mixture of scalars and tensors:

```javascript
var ad = require('adnn/ad');
var Tensor = require('adnn/tensor');

function dot(vec) {
  var sq = ad.tensor.mul(vec, vec);
  return ad.tensor.sumreduce(sq);
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
```

#### Simple neural network

adnn makes it easy to define simple, feedforward neural networks. Here's a basic multilayer perceptron that takes a feature vector as input and outputs class probabilities:

```javascript
var Tensor = require('adnn/tensor');
var ad = require('adnn/ad');
var nn = require('adnn/nn');
var opt = require('adnn/opt');

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

// Train the parameters of the network from some dataset
// 'loadData' is a stand-in for a user-provided function that
//    loads in an array of {input: , output: } objects
// Here, 'input' is a feature vector, and 'output' is a class label
var trainingData = loadData(...);
opt.nnTrain(net, trainingData, opt.classificationLoss, {
  batchSize: 10,
  iterations: 100,
  method: opt.adagrad()
});

// Predict class probabilities for new, unseen features
var features = new Tensor([nInputs]).fillRandom();
var classProbs = net.eval(features);
```

#### Convolutional neural network

adnn includes the building blocks necessary to create convolutional networks. Here is a simple example, adapted from a [ConvNetJS](https://github.com/karpathy/convnetjs) example:

```javascript
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
```

#### Recurrent neural network

adnn is also flexible enough to support recurrent neural networks. Here's an example of a rudimentary RNN:

```javascript
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
  return outputs;
}
```

### The `ad` module 
The `ad` module has its own documentation [here](ad/README.md)

### The `nn` module
The `nn` module has its own documentation [here](nn/README.md)

### The `opt` module
The `opt` module has its own documentation [here](opt/README.md)

### Tensors

adnn provides a `Tensor` type for representing multidimensional arrays of numbers and various operations on them. This is the core datatype underlying neural net computations.

```javascript
var Tensor = require('adnn/tensor');

// Create a rank-1 tensor (i.e. a vector)
var vec = new Tensor([3]);  // vec is a 3-D vector
// Fill vec with the contents of an array
vec.fromArray([1, 2, 3]);
// Return the contents of vec as an array
vec.toArray();  // returns [1, 2, 3]
// Fill vec with a given value
vec.fill(1);    // vec is now [1, 1, 1]
// Fill vec with random values
vec.fillRandom();
// Create a copy of vec
var dupvec = vec.clone();

// Create a rank-2 tensor (i.e. a matrix)
var mat = new Tensor([2, 2]);  // mat is a 2x2 matrix
// Fill mat with the contents of an array
mat.fromArray([[1, 2], [3, 4]);
// Can also use a flattened array
mat.fromFlatArray([1, 2, 3, 4]);
// Retrieve an individual element of mat
var elem = mat.get([0, 1]);   // elem = 2
// Set an individual element of mat
mat.set([0, 1], 5);   // mat is now [[1, 5], [3, 4]]
```

The `Tensor` type also provides a large number of mathematical functions--unary operators, binary operators, reductions, matrix operations, etc. See [tensor.js](tensor.js) for a complete listing.

### Projects using adnn

If you use adnn for anything, let us know and we'll list it here! Send email to daniel.c.ritchie@gmail.com

 - [Neurally-Guided Procedural Models: Learning to Guide Procedural Models with Deep Neural Networks](http://arxiv.org/abs/1603.06143)
 - [WebPPL](https://github.com/probmods/webppl) uses adnn as part of its variational inference implementation.

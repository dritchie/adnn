# The `opt` module

This module provides methods for optimizing functions which take multiple parameters and produce a single output loss (i.e. negative objective) value.
It implements several stochastic gradient optimization methods which can use either automatically-computed or user-provided gradients.
In addition, the module exposes convenience utilities to make common neural net training tasks easier.

### Training neural nets

`opt.nnTrain` is a convenience function for training a single neural net from data:

```javascript
var ad = require('adnn/ad');
var nn = require('adnn/nn');
var opt = require('adnn/opt');

// Train a classifier
// e.g. 20 features -> 5 classes
// data is an array of {input: Tensor([20]), output: Number},
//   where the output number is a class label
var classifier = nn.sequence([
  nn.linear(20, 10),
  nn.tanh,
  nn.linear(10, 5),
  nn.softmax
]);
var data = ...; // Load data here
opt.nnTrain(classifier, data, opt.classificationLoss, {
  batchSize: 10,
  iterations: 1000,
  method: opt.adagrad()
});

// Train a regressor
// e.g. linear function R^20 -> R^5
// data is an array of {input: Tensor([20]), output: Tensor([5])}
var regressor = nn.linear(20, 5);
var data = ...; // Load data here
opt.nnTrain(regressor, data, opt.regressionLoss, {
  batchSize: 10,
  iterations: 1000,
  method: opt.adagrad()
});

// Can use your own loss functions instead of 'classificationLoss' or 'regressionLoss'
// argument 0: the output of the neural net
// argument 1: the 'output' field from a training data point
// Returns an AD node containing the (scalar) loss value
function customLoss(nnOutput, trainingDataOutput) { ... }

// There are several optimization methods available
var optMethods = [
  // stepSizeDecay: multiplicative decay factor on stepSize after each update
  opt.sgd({stepSize: 0.1, stepSizeDecay: 0.99}),
  opt.adagrad({stepSize: 0.1}),
  opt.rmsprop({stepSize: 0.1, decayRate: 0.9}),
  // decayRate1: decay rate for first moment estimate
  // decayRate2: decay rate for second moment estimate
  opt.adam({stepSize: 0.1, decayRate1: 0.9, decayRate2: 0.99})
];

```

### Training AD functions

(if you’re using ad directly, or if the function you want to train isn’t easily expressible as a single neural net)
nnTrain is implemented in terms of this
Show how parameters work
 - ParamStruct = Tensor | array(ParamStruct) | object(ParamStruct)
 - show examples, too
 
### Optimizing AD functions
(if the objective you want to optimize isn’t based on training data)

### Optimizing with user-provided gradients
If you just want to use opt with your own, hand-calculated gradients, that’s cool too
All of the above methods are implemented in terms of this

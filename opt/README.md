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
var data = ...; // Load training data here
opt.nnTrain(classifier, data, opt.classificationLoss, {
  batchSize: 10,
  iterations: 1000,
  method: opt.adagrad(),
  verbose: true   // prints iteration count
});

// Train a regressor
// e.g. linear function R^20 -> R^5
// data is an array of {input: Tensor([20]), output: Tensor([5])}
var regressor = nn.linear(20, 5);
var data = ...; // Load training data here
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

`opt.adTrain` provides a nearly identical interface to `opt.nnTrain`, but for training general AD functions that are not encapsulated in a neural net. In this case, in addition to its output, the function must also return its parameters so that the optimizer knows what free parameters to optimize. `opt.nnTrain` is actually implemented in terms of `opt.adTrain`.

```javascript
var ad = require('adnn/ad');
var opt = require('adnn/opt');

// Learn the parameters of a simple linear function
// (i.e. 'params' dot 'input')
var params = ad.params([10]);
function dot(input) {
  var output = ad.tensor.sumreduce(ad.scalar.mul(input, params));
  return {
    output: output,
    parameters: params
  };
}
var data = ...;   // Load training data here
opt.adTrain(dot, data, opt.regressionLoss, {
  batchSize: 1,
  iterations: 1000,
  method: opt.sgd({stepSize: 0.1})
});

// In the above example, 'parameters' was a single lifted Tensor
// In general, 'parameters' can be an arbitrary structure of the following type:
//    ParamStruct = Tensor | array(ParamStruct) | object(ParamStruct})
// So the following are all valid parameter structures:
var parameters1 = ad.params([20]);
var parameters2 = [ ad.params([5]), ad.params([10]) ];
var parameters3 = { p1: ad.params([5]), p2: ad.params([10]);
var parameters4 = {
  p1: [ ad.params([5]), ad.params([10]) ],
  p2: [ ad.params([6]), ad.params([14]) ]
};
```
 
### Optimizing AD functions
If you want to optimize an objective that isn't based on training data, then you want `opt.adOptimize`.
The function to be optimized must return a loss value (a scalar AD node) as well as its free parameters:

```javascript
var ad = require('adnn/ad');
var opt = require('adnn/opt');

// Find the minimum of some arbitrary function
var params = ad.params([10]);
function foo(input) {
  var output = ad.tensor.sumreduce(
    ad.tensor.exp(ad.tensor.sqrt(ad.tensor.add(input, params)))
  );
  return {
    loss: output,
    parameters: params
  };
}
opt.adOptimize(foo, {
  iterations: 1000,
  method: opt.sgd({stepSize: 0.1})
});
```

### Optimizing with user-provided gradients

It is also possible to use `opt`'s optimization facilities without using AD (i.e. by calculating gradients yourself).
`opt.optimize` does this: it expects a function that returns both its free parameters as well as the gradient of the loss with respect to those parameters. All of the other optimization/training methods in `opt` are implemented in terms of this method.

```javascript
var Tensor = require('adnn/tensor');
var opt = require('adnn/opt');

// Find the minimum of some arbitrary function
var params = new Tensor([10]).fillRandom();
function foo(input) {
  var output = input.add(params).sqrt().exp().sumreduce();
  var gradients = ...;  // Compute gradients here (I'm too lazy... :P)
  return {
    gradients: gradients,
    parameters: params
  };
}
opt.optimize(foo, {
  iterations: 1000,
  method: opt.sgd({stepSize: 0.1})
});
```

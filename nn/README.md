# The `nn` module

In adnn, a neural network is just an AD function with some associated parameters. The `nn` module contains code that makes it easy to define and work with such objects.

### Interacting with neural nets

All neural networks in adnn expose a common interface:

```javascript
var nn = require('adnn/nn');
var Tensor = require('adnn/tensor');

var net = nn.linear(10, 10);

// Feed an input through a network by calling its 'eval' method
var input = new Tensor([10]).fillRandom();
net.eval(input);

// Networks have associated parameters, which is a list of tensors.
net.parameters;   // May be empty for certain networks

// When training is enabled, NNs will compute partial derivatives w.r.t.
//    their parameters. Otherwise, they do not (for efficiency).
net.setTraining(true);

// NNs can also be serialized to JSON objects and deserialized back.
// This saves their structure as well as their current parameter values.
net = nn.deserializeJSON(net.serializeJSON());
```

### Available neural net primitives

adnn comes with many built-in neural nets:

#### Tensor functions
For every function in `ad.tensor`, there is a corresponding neural net in the `nn` module. These are all simple networks with no parameters.

#### Basic layers

```javascript
var nn = require('adnn/nn');

// 'nn.linear(nInputs, nOutputs)' creates a fully-connected layer
var fc = nn.linear(100, 10);

// There are several available nonlinear activation functions
var activations = [
 nn.sigmoid,
 nn.tanh,
 nn.relu
];

// 'nn.mlp' is a convenience function for defining multilayer perceptrons
var mlp = nn.mlp(100  // Number of inputs
  [{nOut: 50, activation: nn.tanh},   // 'activation' is optional
  {nOut: 10}]
);
// mlp: linear(100, 50) -> tanh -> linear(50, 10)
```

#### Convolutional layers

```javascript
var nn = require('adnn/nn');

// 'nn.convolution' creates a convolutional layer
var conv = nn.convolution({
  inDepth:  // Number of input channels (default: 1)
  outDepth: // Number of output channels (default: 1)
  filterSize:    // Width/height of filter
  filterWidth:   // Width of filter (default: filterSize)
  filterHeight:  // Height of filter (default: filterSize)
  stride:   // Number of pixels to skip between filter windows (default: 1)
  strideX:  // Number of horizontal pixels to skip between filter windows (default: stride)
  strideY:  // Number of vertical pixels to skip between filter windows (default: stride)
  pad:  // Number of pixels to pad the output with (default: make output same size as input)
  padX: // Number of horizontal pixels to pad the output with (default: pad)
  padY: // Number of vertical pixels to pad the output with (default: pad)
});

// A number of pooling layers are available; their interfaces are identical
var pools = [
  nn.maxpool,
  nn.minpool,
  nn.meanpool
];
var maxp = nn.maxpool({
  filterSize:    // Width/height of filter
  filterWidth:   // Width of filter (default: filterSize)
  filterHeight:  // Height of filter (default: filterSize)
  stride:   // Number of pixels to skip between filter windows (default: filterSize)
  strideX:  // Number of horizontal pixels to skip between filter windows (default: filterWidth)
  strideY:  // Number of vertical pixels to skip between filter windows (default: filterHeight)
  pad:  // Number of pixels to pad the output with (default: 0)
  padX: // Number of horizontal pixels to pad the output with (default: pad)
  padY: // Number of vertical pixels to pad the output with (default: pad)
});

```

### Creating your own neural nets

There are several different avenues for creating your own neural nets:

#### Networks with no parameters

If you wish to implement a network layer that corresponds to a fixed function with no parameters, implement it as an AD function and then turn it into a NN by calling `nn.lift` on it. This is how all the functions in `ad.tensor` are turned into NNs.

#### Extend `nn.Network`

If you have a network with complex internal logic that cannot be captured by the composition of existing networks, then you can extend the `nn.Network` class (see [network.js](network.js)). Be sure to provide an implementation of `setTraining` that does the right thing, and make all parameters available in the `parameters` member. You'll also need to implement `serializeJSON`:
```javascript
var nn = require('adnn/nn');

// For a new NN class called 'MyNetwork'
MyNetwork.prototype.serializeJSON = function() {
  return {
    type: 'mynetwork',
    // other fields here...
  };
};

// Declare a deserializer corresponding to the 'type' field used above
nn.Network.deserializers.mynetwork = function(json) {
  // parse JSON object and return new MyNetwork object
};
```

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

adnn also supports computations involving tensors:


#### Simple neural network ####

#### Convolutional neural network ####

#### Recurrent neural network ####

### The `ad` module ###

### The `nn` module ###

### Tensors ###

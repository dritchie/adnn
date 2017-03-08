# The `ad` module
The `ad` module implements [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation). Computations using AD functions implicitly build a graph of all operations. When the computation finishes, the graph can be walked backward to accumulate partial derivatives from the output back to the inputs.

### Interacting with AD

There is a simple interface for interacting with AD functions and AD values:

```javascript
var ad = require('adnn/ad');

// Raw Numbers/Tensors can be used with AD functions
ad.scalar.tanh(1.5);  // 0.9051...
ad.tensor.tanh(new Tensor([3]).fill(1.5));  // [0.9051, 0.9051, 0.9051]

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

// It's also possible to check whether a value is a lifted AD Node or not
ad.isLifted(scalarIn);  // true
ad.isLifted(1.5);       // false
```

### Available AD primitive functions

adnn comes with a large number of built-in AD primitives:
- **Unary operators**
  - Defined for both scalars (in `ad.scalar`) and tensors (in `ad.tensor`)
  - floor, ceil, round, sqrt, exp, log, abs, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, sigmoid
- **Binary operators**
  - Defined for both scalars (in `ad.scalar`) and tensors (in `ad.tensor`)
  - add, sub, mul, div, pow, min, max, atan2
- **Comparators**
  - Currently defined for scalars only (in `ad.scalar`)
  - eq (==), neq (!=), peq (===), pneq (!==), gt (>), lt (<), geq (>=), leq (<=)
- **Miscellaneous**
  - **`ad.tensorEntry(x, i)`**: Extracts the `i`th element of `x` and returns it as a scalar.
  - **`ad.tensorToScalars(x)`**: Turns tensor `x` into a list of scalars.
  - **`ad.scalarsToTensor(lst)`**: Turns a list of scalars `lst` into a tensor.
  - **`ad.sumreduce(x)`**: Returns the sum of entries of a tensor `x`.
  - **`ad.scalar.sum(lst)`**: Returns the sum of the list of scalars `lst`.
  - **`ad.tensor.range(x, i, j)`**: Returns a tensor constructed from elements `i` through `j` (non-inclusive) of tensor `x`.
  - **`ad.tensor.split(x, sizes)`**: Split tensor `x` into `sizes.length` tensors, where the size of the output tensors are given by `sizes`.
  - **`ad.tensor.concat(lst)`**: Concatenate a list of tensors `lst` into one tensor.
  - **`ad.tensor.softmax(x)`**: Compute the [Softmax](https://en.wikipedia.org/wiki/Softmax_function) function for a tensor `x`.

For more information, see [functions.js](functions.js).

### Defining new primitives

adnn also provides an interface for creating your own AD primitive functions:

```javascript
var ad = require('adnn/ad');

// Defining unary functions
var newUnaryFn = ad.newUnaryFunction({
  OutputType: // Either Number or Tensor
  name:       // The name of the new function
  forward: function(x) {...}  // Implements the function
  backward: function(xnode) {...} // Accumulates into derivative(xnode). Output node available as 'this'
});

// Defining binary functions
var newBinaryFn = ad.newBinaryFunction({
  OutputType: // Either Number or Tensor
  name:       // The name of the new function
  forward: function(x, y) {...}  // Implements the function
  backward1: function(xnode, y) {...} // Accumulates into derivative(xnode). Output node available as 'this'
  backward2: function(x, ynode) {...} // Accumulates into derivative(ynode). Output node available as 'this'
});

// Defining arbitrary functions
var newUnaryFn = ad.newUnaryFunction({
  OutputType: // Either Number or Tensor
  name:       // The name of the new function
  forward: function(...) {...}  // Implements the function
  backward: function(...) {...} // Accumulates into derivatives of all Node inputs. Output node available as 'this'
  getParents: function(...) {...} // Returns a list of inputs which are Nodes.
});

// Can also 'lift' non-differentiable functions to operate on Nodes
var nan = ad.liftUnaryFunction(isNaN);
var eq = ad.liftBinaryFunction(function(x, y) { return x == y; });
```
For more information, see [func.js](func.js).

### Macro transforms for scalar code

Code which uses scalar math functions can be automatically converted to use scalar AD functions via a [Sweet.js](http://sweetjs.org/) macro transform (see [macros.sjs](macros.sjs)). There are several different ways to accomplish this:

#### Using node via command line / REPL
In this setting, the simplest way to use the macro transform is via the `ad.macroRequire` function:
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
var dist = ad.macroRequire('./dist.js'); // 'dist' is now an AD function
```
See [transform.js](transform.js) to learn more about this function.

#### Transforming code in the browser

Currently, `ad.macroRequire` is not available in the browser, as attempting to load a [browserified](http://browserify.org/) script which includes Sweet.js will throw an error. For the time being, one workaround is to directly use Sweet.js to macro-transform your code (see their FAQ section on [How to run Sweet.js in the browser](http://sweetjs.org/doc/main/sweet.html#how-do-i-run-sweet.js-in-the-browser)).

#### Pre-compiling macro code for the browser

If you wish to include macro transformation as part of a compile / package / minify pipeline for creating a browser script, then check out the [sweetify](https://github.com/andreypopp/sweetify) transform plugin for browserify.

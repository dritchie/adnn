'use strict';

var graph = require('./graph.js');
var Node = graph.Node;
var func = require('./func.js');
var derivs = require('./adjs/derivatives.js');
var fn = require('./functions.js')
var fns = fn.fns;

var Scalar = Number;

Math.sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };

function makeScalarFunctions() {

    // Define which backwards derivatives we'll use for the given OutputType
    function backward(derivFns) {
        return derivFns.scalar;
    }

    var namePrefix = 'scalar.';

    // Lifted unary operators
    var unops = {
        neg: function(x) { return -x; }
    };
    for (var op in unops) {
        fns[op] = func.newUnaryFunction({
            OutputType: OutputType,
            name: namePrefix+op,
            forward: unops[op],
            backward: backward(derivs[op])
        });
    }

    // Lifted binary operators
    var binops = {
        add: function(x, y) { return x + y; },
        sub: function(x, y) { return x - y; },
        mul: function(x, y) { return x * y; },
        div: function(x, y) { return x / y; }
    };
    for (var op in binops) {
        fns[op] = func.newBinaryFunction({
            OutputType: OutputType,
            name: namePrefix+op,
            forward: binops[op],
            backward1: backward(derivs[op])[0],
            backward2: backward(derivs[op])[1]
        });
    }

    // Lifted Math functions
    var unaryFns = [
        'floor', 'ceil', 'round', 'sqrt', 'exp', 'log', 'abs', 'sin', 'cos',
        'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh',
        'acosh', 'atanh', 'sigmoid'
    ];
    var binaryFns = [
        'pow', 'min', 'max', 'atan2'
    ];
    for (var i = 0; i < unaryFns.length; i++) {
        var fnname = unaryFns[i];
        var forward = new Function('x', 'return Math.' + fnname + '(x);');
        fns[fnname] = func.newUnaryFunction({
            OutputType: OutputType,
            name: namePrefix+fnname,
            forward: forward,
            backward: backward(derivs[fnname]),
        });
    }
    for (var i = 0; i < binaryFns.length; i++) {
        var fnname = binaryFns[i];
        var forward = new Function('x', 'y', 'return Math.' + fnname + '(x, y);');
        fns[fnname] = func.newBinaryFunction({
            OutputType: OutputType,
            name: namePrefix+fnname,
            forward: forward,
            backward1: backward(derivs[fnname])[0],
            backward2: backward(derivs[fnname])[1]
        });
    }

    // NaN and infinity checks
    fns.isNaN = func.liftUnaryFunction(isNaN);
    fns.isFinite = func.liftUnaryFunction(isFinite);

    return fns;
}


fns[scalar] = makeScalarFunctions();

// Re-export Math constants etc.
Object.getOwnPropertyNames(Math).forEach(function(p) {
  if (!fns.scalar.hasOwnProperty(p)) {
    fns.scalar[p] = Math[p];
  }
});

// Sum an arbitrary number of scalars
// Can either take an array of scalars or a variable number of arguments
fns.scalar.sum = func.newFunction({
    OutputType: Scalar,
    name: 'scalar.sum',
    forward: function() {
        var args = arguments.length === 1 && arguments[0] instanceof Array ?
            arguments[0] : arguments;
        var thesum = 0;
        var n = args.length;
        while (n--) {
            var arg = args[n];
            var x = arg instanceof Node ? arg.x : arg;
            thesum += x;
        }
        return thesum;
    },
    backward: function() {
        var args = arguments.length === 1 && arguments[0] instanceof Array ?
            arguments[0] : arguments;
        var n = args.length;
        while (n--) {
            var arg = args[n];
            if (arg instanceof Node) {
                arg.dx += this.dx;
            }
        }
    },
    getParents: func.naryGetParents
});


fns.scalar.eq = func.liftBinaryFunction(
    function(x, y) { return x == y; }
);

fns.scalar.neq = func.liftBinaryFunction(
    function(x, y) { return x != y; }
);

fns.scalar.peq = func.liftBinaryFunction(
    function(x, y) { return x === y; }
);

fns.scalar.pneq = func.liftBinaryFunction(
    function(x, y) { return x !== y; }
);

fns.scalar.gt = func.liftBinaryFunction(
    function(x, y) { return x > y; }
);

fns.scalar.lt = func.liftBinaryFunction(
    function(x, y) { return x < y; }
);

fns.scalar.geq = func.liftBinaryFunction(
    function(x, y) { return x >= y; }
);

fns.scalar.leq = func.liftBinaryFunction(
    function(x, y) { return x <= y; }
);

module.exports = fns;
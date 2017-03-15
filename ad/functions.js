'use strict';

var TH = require('../THTensor.js');
var T = require('../tensor.js');
var func = require('./func.js');
var derivs = require('./adjs/derivatives.js');
var thderivs = require('./adTH/derivatives.js');

var fns = {};

function makeFunctions(OutputType) {

    // Define which backwards derivatives we'll use for the given OutputType
    function backward(derivFns) {
        return OutputType === Tensor ? derivFns.tensor : THTensor ? derivFns.thtensor : derivFns.scalar;
    }

    var namePrefix = OutputType === Scalar ? 'scalar.' : OutputType === Tensor ? 'tensor.' : 'thtensor.';

    // Lifted unary operators
    var unops = {
        neg: (OutputType === Tensor || OutputType === THTensor) ?
            function(x) { return x.neg(); } :
            function(x) { return -x; }
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
        add: (OutputType === Tensor || OutputType === THTensor) ?
            function(x, y) { return x.add(y); } :
            function(x, y) { return x + y; },
        sub: (OutputType === Tensor || OutputType === THTensor) ?
            function(x, y) { return x.sub(y); } :
            function(x, y) { return x - y; },
        mul: (OutputType === Tensor || OutputType === THTensor) ?
            function(x, y) { return x.mul(y); } :
            function(x, y) { return x * y; },
        div: (OutputType === Tensor || OutputType === THTensor) ?
            function(x, y) { return x.div(y); } :
            function(x, y) { return x / y; }
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
        var forward = (OutputType === Tensor || OutputType === THTensor) ?
            new Function('x', 'return x.' + fnname + '();') :
            new Function('x', 'return Math.' + fnname + '(x);');
        fns[fnname] = func.newUnaryFunction({
            OutputType: OutputType,
            name: namePrefix+fnname,
            forward: forward,
            backward: backward(derivs[fnname]),
        });
    }
    for (var i = 0; i < binaryFns.length; i++) {
        var fnname = binaryFns[i];
        var forward = (OutputType === Tensor || OutputType === THTensor) ?
            new Function('x', 'y', 'return x.' + fnname + '(y);') :
            new Function('x', 'y', 'return Math.' + fnname + '(x, y);');
        fns[fnname] = func.newBinaryFunction({
            OutputType: OutputType,
            name: namePrefix+fnname,
            forward: forward,
            backward1: backward(derivs[fnname])[0],
            backward2: backward(derivs[fnname])[1]
        });
    }

    // NaN and infinity checks
    fns.isNaN = OutputType === Scalar ?
        func.liftUnaryFunction(isNaN) :
        func.liftUnaryFunction(function(t) { return t.isNaN(); });
    fns.isFinite = OutputType === Scalar ?
        func.liftUnaryFunction(isFinite) :
        func.liftUnaryFunction(function(t) { return t.isFinite(); });

    return fns;
}


var fns = {
    scalar: makeFunctions(Scalar),
    tensor: makeFunctions(Tensor),
    thtensor: makeFunctions(THTensor)
};

// Re-export Math constants etc.
Object.getOwnPropertyNames(Math).forEach(function(p) {
  if (!fns.scalar.hasOwnProperty(p)) {
    fns.scalar[p] = Math[p];
  }
});

module.exports = fns;




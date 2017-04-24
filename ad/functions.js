'use strict';

var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');
var func = require('./func.js');
var derivs = require('./adjs/derivatives.js');
var thderivs = require('./adTH/derivatives.js');


function makeTensorFunctions(OutputType) {
    var fns = {};

    // Define which backwards derivatives we'll use for the given OutputType
    function backward(derivFns) {
        return derivFns.tensor;
    }

    var namePrefix = OutputType === Tensor ? 'tensor.' : 'thtensor.';

    // Lifted unary operators
    var unops = {
        neg: function(x) { return x.neg(); }
    };
    for (var op in unops) {
        fns[op] = (OutputType === Tensor) ?
            func.newUnaryFunction({
                OutputType: OutputType,
                name: namePrefix+op,
                forward: unops[op],
                backward: backward(derivs[op])
            }) :
            func.newUnaryFunction({
                OutputType: OutputType,
                name: namePrefix+op,
                forward: unops[op],
                backward: backward(thderivs[op])
            });
    }

    // Lifted binary operators
    var binops = {
        add: function(x, y) { return x.add(y); },
        sub: function(x, y) { return x.sub(y); },
        mul: function(x, y) { return x.mul(y); },
        div: function(x, y) { return x.div(y); }
    };
    for (var op in binops) {
        fns[op] = (OutputType === Tensor) ?
            func.newBinaryFunction({
                OutputType: OutputType,
                name: namePrefix+op,
                forward: binops[op],
                backward1: backward(derivs[op])[0],
                backward2: backward(derivs[op])[1]
            }) :
            func.newBinaryFunction({
                OutputType: OutputType,
                name: namePrefix+op,
                forward: binops[op],
                backward1: backward(thderivs[op])[0],
                backward2: backward(thderivs[op])[1]
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
        var forward = new Function('x', 'return x.' + fnname + '();');
        fns[fnname] = (OutputType === Tensor) ?
            func.newUnaryFunction({
                OutputType: OutputType,
                name: namePrefix+fnname,
                forward: forward,
                backward: backward(derivs[fnname]),
            }) : 
            func.newUnaryFunction({
                OutputType: OutputType,
                name: namePrefix+fnname,
                forward: forward,
                backward: backward(thderivs[fnname]),
            });
    }
    for (var i = 0; i < binaryFns.length; i++) {
        var fnname = binaryFns[i];
        var forward = new Function('x', 'y', 'return x.' + fnname + '(y);');
        fns[fnname] = (OutputType === Tensor) ?
            func.newBinaryFunction({
                OutputType: OutputType,
                name: namePrefix+fnname,
                forward: forward,
                backward1: backward(derivs[fnname])[0],
                backward2: backward(derivs[fnname])[1]
            }) :
            func.newBinaryFunction({
                OutputType: OutputType,
                name: namePrefix+fnname,
                forward: forward,
                backward1: backward(thderivs[fnname])[0],
                backward2: backward(thderivs[fnname])[1]
            });
    }

    // NaN and infinity checks
    fns.isNaN = func.liftUnaryFunction(function(t) { return t.isNaN(); });
    fns.isFinite = func.liftUnaryFunction(function(t) { return t.isFinite(); });

    return fns;
}


var fns = {
    // scalar: makeFunctions(Scalar),
    tensor: makeTensorFunctions(Tensor),
    thtensor: makeTensorFunctions(THTensor)
};

// Re-export Math constants etc.
// Object.getOwnPropertyNames(Math).forEach(function(p) {
//   if (!fns.scalar.hasOwnProperty(p)) {
//     fns.scalar[p] = Math[p];
//   }
// });

module.exports = fns


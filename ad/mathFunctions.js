'use strict';

var Tensor = require('../tensor.js');
var THTensor = require('../THTensor.js'); // tensor = tensor
var mathFns = require('./functions.js');
var graph = require('./graph.js');
var Node = graph.Node;
var func = require('./func.js');

var fns = {tensor: {}};

// Wrapper around functions.js to handle JS vs TH tensors

var unaryFns = [
    'neg', 'floor', 'ceil', 'round', 'sqrt', 'exp', 'log', 'abs', 'sin',
    'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh',
    'acosh', 'atanh', 'sigmoid'
];
var binaryFns = [
    'add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'atan2'
];

function getUnaryFunction(fname) {
    return function (t) {
        var ten = t instanceof Node ? t.x : t;
        if (ten instanceof Tensor)
            return mathFns.tensor[fname](t);
        return mathFns.thtensor[fname](t);
    }
}

function getBinaryFunction(fname) {
    return function (x, y) {
        var x_l = x instanceof Node ? x.x : x;
        var y_l = y instanceof Node ? y.x : y;
        if (x_l instanceof Tensor)
            return mathFns.tensor[fname](x, y);
        return mathFns.thtensor[fname](x, y);
    }
}

fns.tensor.isNaN = func.liftUnaryFunction(function(t) { return t.isNaN(); });

fns.tensor.isFinite = func.liftUnaryFunction(function(t) { return t.isFinite(); });

for (var i = 0; i < unaryFns.length; i++) {
    var fnname = unaryFns[i];
    fns.tensor[fnname] = getUnaryFunction(fnname);
}

for (var i = 0; i < binaryFns.length; i++) {
    var fnname = binaryFns[i];
    // console.log(fnname)
    fns.tensor[fnname] = getBinaryFunction(fnname)
}

module.exports = fns;
'use strict';

var Tensor = require('../tensor.js');
var THTensor = require('../THTensor.js'); // tensor = tensor
var jslaFunc = require('./adjs/linalgFunctions.js')
var thlaFunc = require('./adTH/linalgFunctions.js')
var graph = require('./graph.js');
var Node = graph.Node;
var func = require('./func.js');
//var derivs = require('./derivatives.js');
var _ = require('lodash')

var fns = {tensor: {}};

fns.tensor.transpose = function (t) {
    var ten = t instanceof Node ? t.x : t;
    if (ten instanceof Tensor)
        return jslaFunc.tensor.transpose(t);
    return thlaFunc.thtensor.transpose(t);
}

fns.tensor.diagonal = function (t) {
    var ten = t instanceof Node ? t.x : t;
    if (ten instanceof Tensor)
        return jslaFunc.tensor.diagonal(t);
    return thlaFunc.thtensor.diagonal(t);
}

fns.tensor.inverse = function (t) {
    var ten = t instanceof Node ? t.x : t;
    if (ten instanceof Tensor)
        return jslaFunc.tensor.inverse(t);
    return thlaFunc.thtensor.inverse(t);
}

fns.tensor.determinant = function (t) {
    var ten = t instanceof Node ? t.x : t;
    if (ten instanceof Tensor)
        return jslaFunc.tensor.determinant(t);
    return thlaFunc.thtensor.determinant(t);
}

fns.tensor.dot = function (a, b) {
    var A = a instanceof Node ? a.x : a;
    var B = b instanceof Node ? b.x : b;
    if (A instanceof Tensor)
        return jslaFunc.tensor.dot(A, B);
    return thlaFunc.thtensor.dot(A, B);
}


module.exports = fns;

'use strict';

var utils = require('../utils.js');
var graph = require('./graph.js');
var Tensor = require('../tensor.js');
var THTensor = require('../THTensor.js');

var Node = graph.Node;
var ScalarNode = graph.ScalarNode;
var TensorNode = graph.TensorNode;
var THTensorNode = graph.THTensorNode;

var emptylist = [];

function liftScalar(x, name) { return new ScalarNode(x, emptylist, emptylist, undefined, name); };
function liftTensor(x, name) {
  if (x instanceof Tensor)
    return new TensorNode(x, emptylist, emptylist, undefined, name);
  return new THTensorNode(x, emptylist, emptylist, undefined, name);
};

function doLift(x, name) {
    return (x instanceof Tensor || x instanceof THTensor) ? liftTensor(x, name) : liftScalar(x, name);
}

var ad = {
    lift: function(x, name) { return x instanceof Node ? x : doLift(x, name); },
    isLifted: function(x) { return x instanceof Node; },
    value: function(x) { return x instanceof Node ? x.x : x; },
    derivative: function(x) { return x.dx; },
};

// Create randomly-initialized params
// TODO: Use orthogonal initialization?
ad.params = function(dims, name) {
    return ad.lift(new Tensor(dims).fillRandom(), name); 
};

var func = require('./func.js');
var functions = require('./functions.js').fns;
var tenFunctions = require('./tensorFunctions.js');
var linalgFunctions = require('./linalgFunctions.js');
var scalars = require('./scalar.js');
utils.mergeObjects(functions.tensor, tenFunctions.tensor, linalgFunctions.tensor);
ad = utils.mergeObjects(ad, func, functions, scalars);

module.exports = ad;
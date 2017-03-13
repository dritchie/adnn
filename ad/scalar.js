'use strict';

var Tensor = require('../THTensor.js');
var graph = require('./graph.js');
var Node = graph.Node;
var func = require('./func.js');
var derivs = require('./adjs/derivatives.js');
var fn = require('./functions.js')
var fns = fn.fns;

var Scalar = Number;

Math.sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };

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
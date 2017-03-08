'use strict';

var Tensor = require('../THTensor.js');
var graph = require('./adjs/graph.js');
var Node = graph.Node;
var func = require('./adjs/func.js');
var derivs = require('./adjs/derivatives.js');


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

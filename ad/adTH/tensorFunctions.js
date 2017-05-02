'use strict';

var THTensor = require('../../THTensor.js'); // tensor = THTensor
var graph = require('../graph.js');
var Node = graph.Node;
var func = require('../func.js');
var derivs = require('./derivatives.js');
var _ = require('lodash')

var fns = {thtensor: {}};

// Tensor reductions  -----------------------------------------------------

var Scalar = Number

fns.thtensor.sumreduce = func.newUnaryFunction({
    OutputType: Scalar,
    name: 'sumreduce',
    forward: function(t) {
        return t.sumreduce();
    },
    backward: function(t) {
        var n = t.dx.length;
        while (n--) {
            t.dx.data[n] += this.dx;
        }
    }
});

fns.thtensor.allreduce = func.liftUnaryFunction(function(t) {
    return t.allreduce();
});

fns.thtensor.anyreduce = func.liftUnaryFunction(function(t) {
    return t.anyreduce();
});

// TODO: min/max?


// Scalar/tensor shaping operations ---------------------------------------


// Select one entry out of a tensor (by linear indexing)
fns.thtensor.get = func.newFunction({
    OutputType: Scalar,
    name: 'thtensor.get',
    forward: function(t, i) {
        return t instanceof Node ? t.x.get([i]) : t.get([i]);
    },
    backward: function(t, i) {
        if (t instanceof Node) {
            t.dx.data[i] += this.dx;
        }
    },
    getParents: function(t, i) {
        return t instanceof Node ? [t] : [];
    }
});

// Split a tensor into an array of its scalar entries
fns.thtensor.toScalars = function(t) {
    var n = t instanceof Node ? t.x.length : t.length;
    var s = new Array(n);
    while (n--) {
        s[n] = fns.thtensor.get(t, n);
    }
    return s;
};

// Select a subtensor from a larger tensor
// TODO: Eventually implement this as a view into existing storage,
//    probably using refClone (+ other new stuff)
fns.thtensor.range = func.newFunction({
    OutputType: THTensor,
    name: 'thtensor.range',
    forward: function(t, start, end) {
        return t.range(start, end);
    },
    backward: function(t, start, end) {
        if (t instanceof Node) {
            var n = end - start;
            while (n--) {
                var i = start + n;
                t.dx.data[i] += this.dx.data[n];
            }
        }
    },
    getParents: function(t, start, end) {
        return t instanceof Node ? [t] : [];
    }
});


// Split a tensor into multiple smaller tensors
fns.thtensor.split = function(t, lengths) {
    var ts = new Array(lengths.length);
    var start = 0;
    for (var i = 0; i < lengths.length; i++) {
        var l = lengths[i];
        ts[i] = fns.thtensor.range(t, start, start + l);
        start += l;
    }
    return ts;
};

// Concatentate multiple scalars into a tensor
// Can either take an array of scalars or a variable number of arguments
fns.thtensor.fromScalars = func.newFunction({
    OutputType: THTensor,
    name: 'thtensor.fromScalars',
    forward: function() {
        var args = arguments.length === 1 && arguments[0] instanceof Array ?
            arguments[0] : arguments;
        var n = args.length;
        var t = new THTensor([n]);
        var val = args instanceof Node ? args.x : args;
        t.fromArray(val)
        // while (n--) {
        //     var arg = args[n];
        //     var val = arg instanceof Node ? arg.x : arg;
        //     t.set([n], val);
        // }
        return t;
    },
    backward: function() {
        var args = arguments.length === 1 && arguments[0] instanceof Array ?
            arguments[0] : arguments;
        var n = args.length;
        while (n--) {
            var arg = args[n];
            if (arg instanceof Node) {
                arg.dx += this.dx.data[n];
            }
        }
    },
    getParents: func.naryGetParents
});

// Concatentate multiple tensors into one big tensor
// Can either take an array of tensors or a variable number of arguments
// TODO: Eventually implement this as views into multiple storages?
// No offset copying, so right now this implementation is working but slow
fns.thtensor.concat = func.newFunction({
    OutputType: THTensor,
    name: 'thtensor.concat',
    forward: function() {
        var args = arguments[0]
        var t = args[0] instanceof Node ? args[0].x : args[0];
        var out = t.concat([args[1]]);
        return out;
        // var args = arguments.length === 1 && arguments[0] instanceof Array ?
        //     arguments[0] : arguments;
        // var n = args.length;
        // var size = 0;
        // while (n--) {
        //     var arg = args[n];
        //     var tn = arg instanceof Node ? arg.x : arg;
        //     size += tn.length;
        // }
        // var t = new THTensor([size]);
        // var i = 0
        // _.forEach(args, function(v) {
        //     var tn = v instanceof Node ? v.x : v;
        //     for (var j = 0; j < tn.length; j++) {
        //        t.set([i++], tn.get([j]));
        //     }
        // });
        // return t;
    },
    backward: function() {
        var args = arguments.length === 1 && arguments[0] instanceof Array ?
            arguments[0] : arguments;
        var n = args.length;
        var i = 0;
        for (var j = 0; j < n; j++) {
            var arg = args[j];
            if (arg instanceof Node) {
                var tn = arg;
                var len = tn.dx.length;
                while (len--) {
                    tn.dx.data[len] += this.dx.data[i + len];
                }
                i += tn.dx.length;
            } else i += arg.length;
        }
    },
    getParents: func.naryGetParents
});

// Reshape a tensor
// Creates a new TensorNode whose x and dx fields are refClones
//    of the corresponding fields on the input node.
fns.thtensor.reshape = function(t, dims) {
    if (t instanceof Node) {
        var node = t.refClone();
        node.x.reshape(dims);
        node.dx.reshape(dims);
        return node;
    } else {
        var ref = t.refClone();
        ref.reshape(dims);
        return ref;
    }
};

// http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
fns.thtensor.softmax = func.newUnaryFunction({
    OutputType: THTensor,
    name: 'thtensor.softmax',
    forward: function(t) {
        return t.softmax();
    },
    backward: function(t) {
        // For each input entry, accumulate partial derivatives
        //    for each output entry
        var n = t.dx.data.length;
        var s = 0;
        for (var i = 0; i < n; i++) {
            s += this.x.data[i] * this.dx.data[i];
        }
        for (var j = 0; j < n; j++) {
            t.dx.data[j] += this.x.data[j] * (this.dx.data[j] - s);
        }
    }
});

module.exports = fns;
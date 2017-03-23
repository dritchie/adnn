'use strict';

var Tensor = require('../../tensor.js');
var graph = require('../graph.js');
var Node = graph.Node;
var func = require('../func.js');

var fns = {tensor: {}};

var Scalar = Number

fns.tensor.sumreduce = func.newUnaryFunction({
    OutputType: Scalar,
    name: 'sumreduce',
    forward: function(t) {
        return t.sumreduce();
    },
    backward: function(t) {
        var n = t.dx.data.length;
        while (n--) {
            t.dx.data[n] += this.dx;
        }
    }
});

fns.tensor.allreduce = func.liftUnaryFunction(function(t) {
    return t.allreduce();
});

fns.tensor.anyreduce = func.liftUnaryFunction(function(t) {
    return t.anyreduce();
});


// Select one entry out of a tensor (by linear indexing)
fns.tensor.get = func.newFunction({
	OutputType: Scalar,
	name: 'tensor.get',
	forward: function(t, i) {
		return t instanceof Node ? t.x.data[i] : t.data[i];
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
fns.tensor.toScalars = function(t) {
	var n = t instanceof Node ? t.x.length : t.length;
	var s = new Array(n);
	while (n--) {
		s[n] = fns.tensor.get(t, n);
	}
	return s;
};

// Select a subtensor from a larger tensor
// TODO: Eventually implement this as a view into existing storage,
//    probably using refClone (+ other new stuff)
fns.tensor.range = func.newFunction({
	OutputType: Tensor,
	name: 'tensor.range',
	forward: function(t, start, end) {
		t = t instanceof Node ? t.x : t;
		var n = end - start;
		var tn = new Tensor([n]);
		while (n--) {
			var i = start + n;
			tn.data[n] = t.data[i];
		}
		return tn;
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
fns.tensor.split = function(t, lengths) {
	var ts = new Array(lengths.length);
	var start = 0;
	for (var i = 0; i < lengths.length; i++) {
		var l = lengths[i];
		ts[i] = fns.tensor.range(t, start, start + l);
		start += l;
	}
	return ts;
};

// Concatentate multiple scalars into a tensor
// Can either take an array of scalars or a variable number of arguments
fns.tensor.fromScalars = func.newFunction({
	OutputType: Tensor,
	name: 'tensor.fromScalars',
	forward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		var t = new Tensor([n]);
		while (n--) {
			var arg = args[n];
			t.data[n] = arg instanceof Node ? arg.x : arg;
		}
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
fns.tensor.concat = func.newFunction({
	OutputType: Tensor,
	name: 'tensor.concat',
	forward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		var size = 0;
		while (n--) {
			var arg = args[n];
			var tn = arg instanceof Node ? arg.x : arg;
			size += tn.length;
		}
		var t = new Tensor([size]);
		n = args.length;
		var i = 0;
		for (var j = 0; j < n; j++) {
			var arg = args[j];
			var tn = arg instanceof Node ? arg.x : arg;
			t.copy(tn, i);
			i += tn.length;
		}
		return t;
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
fns.tensor.reshape = function(t, dims) {
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
fns.tensor.softmax = func.newUnaryFunction({
	OutputType: Tensor,
	name: 'tensor.softmax',
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




'use strict';

var Tensor = require('../tensor.js');
var graph = require('./graph.js');
var Node = graph.Node;
var func = require('./func.js');
var derivs = require('./derivatives.js');


var Scalar = Number;

// Additional scalar functions 'missing' from Math
Math.sigmoid = function(x) { return 1 / (1 + Math.exp(-x)); };


// Scalar & tensor operators and math functions -------------------------------

function makeFunctions(OutputType) {

	var fns = {};

	// Define which backwards derivatives we'll use for the given OutputType
	function backward(derivFns) {
		return OutputType === Tensor ? derivFns.tensor : derivFns.scalar;
	}

	var namePrefix = OutputType === Scalar ? 'scalar.' : 'tensor.';

	// Lifted unary operators
	var unops = {
		neg: OutputType === Tensor ?
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
		add: OutputType === Tensor ?
			function(x, y) { return x.add(y); } :
			function(x, y) { return x + y; },
		sub: OutputType === Tensor ?
			function(x, y) { return x.sub(y); } :
			function(x, y) { return x - y; },
		mul: OutputType === Tensor ?
			function(x, y) { return x.mul(y); } :
			function(x, y) { return x * y; },
		div: OutputType === Tensor ?
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
		var forward = OutputType === Tensor ?
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
		var forward = OutputType === Tensor ?
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
	tensor: makeFunctions(Tensor)
};

// Re-export Math constants etc.
Object.getOwnPropertyNames(Math).forEach(function(p) {
  if (!fns.scalar.hasOwnProperty(p)) {
    fns.scalar[p] = Math[p];
  }
});


// Also lift scalar comparators -----------------------------------------------

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


// Matrix operations  -----------------------------------------------------


fns.tensor.transpose = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'transpose',
  forward: function(a) {
    return a.transpose();
  },
  backward: function(a) {
    var h = this.x.dims[0];
    var w = this.x.dims[1];
    for (var i = 0; i < h; i++) {
      for (var j = 0; j < w; j++) {
        a.dx.data[j * h + i] += this.dx.data[i * w + j];
      }
    }
  }
});

fns.tensor.diagonal = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'diagonal',
  forward: function(a) {
    return a.diagonal();
  },
  backward: function(a) {
    var n = a.dx.dims[0];
    for (var i = 0; i < n; i++) {
      a.dx.data[i] += this.dx.data[i * (n + 1)];
    }
  }
});

fns.tensor.inverse = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'inverse',
  forward: function(A) {
    return A.inverse();
  },
  backward: function(A) {
    var xT = this.x.T();
    A.dx = A.dx.add(xT.dot(this.dx).dot(xT).neg());
  }
});

fns.tensor.determinant = func.newUnaryFunction({
  OutputType: Number,
  name: 'determinant',
  forward: function(A) {
    return A.determinant();
  },
  backward: function(A) {
    // A is square matrix.
    // Assume A is invertable.
    var n = A.x.dims[0];
    var invA = A.x.inv();
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        A.dx.data[i * n + j] += this.x * this.dx * invA.data[j * n + i];
      }
    }
  }
});

fns.tensor.dot = func.newBinaryFunction({
  OutputType: Tensor,
  name: 'dot',
  forward: function(a, b) {
    return a.dot(b);
  },
  backward1: function(A, B) {
    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bw = Bp.dims[1];
    var wout = Bw;

    for (var l = 0; l < Ah; l++) {
      for (var m = 0; m < Aw; m++) {
        var z = 0;
        for (var j = 0; j < wout; j++) {
          z += this.dx.data[l * wout + j] * Bp.data[m * Bw + j];
        }
        A.dx.data[l * Aw + m] += z;
      }
    }
  },
  backward2: function(A, B) {
    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var wout = Bw;

    for (var l = 0; l < Bh; l++) {
      for (var m = 0; m < Bw; m++) {
        var z = 0;
        for (var i = 0; i < Ah; i++) {
          z += this.dx.data[i * wout + m] * Ap.data[i * Aw + l];
        }
        B.dx.data[l * Bw + m] += z;
      }
    }

  }
});


// Tensor reductions  -----------------------------------------------------


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

// TODO: min/max?


// Scalar/tensor shaping operations ---------------------------------------


// Select one entry out of a tensor (by linear indexing)
fns.tensorEntry = func.newFunction({
	OutputType: Scalar,
	name: 'tensorEtry',
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
fns.tensorToScalars = function(t) {
	var n = t instanceof Node ? t.x.length : t.length;
	var s = new Array(n);
	while (n--) {
		s[n] = fns.tensorEntry(t, n);
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
fns.scalarsToTensor = func.newFunction({
	OutputType: Tensor,
	name: 'scalarsToTensor',
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


// Misc. ----------------------------------------------------------------------


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
		for (var j = 0; j < n; j++) {
			var out_j = this.x.data[j];
			for (var i = 0; i < n; i++) {
				var out_i = this.x.data[i];
				var d = out_i * ((i === j) - out_j);
				t.dx.data[j] += d * this.dx.data[i];
			}
		}
	}
});


module.exports = fns;




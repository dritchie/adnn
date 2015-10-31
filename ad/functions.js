var Tensor = require('../tensor.js');
var graph = require('./graph.js');
var func = require('./func.js');
var derivs = require('./derivatives.js');


var Scalar = Number;


// Scalar & tensor operators and math functions -------------------------------

function makeFunctions(OutputType) {

	var fns = {};

	// Define which backwards derivatives we'll use for the given OutputType
	function backward(derivFns) {
		return OutputType === Tensor ? derivFns.tensor : derivFns.scalar;
	}

	// Lifted operators
	var ops = {
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
	for (var op in ops) {
		fns[op] = func.newBinaryFunction(OutputType, {
			forward: ops[op],
			backward1: backward(derivs[op])[0],
			backward2: backward(derivs[op])[1]
		});
	}

	// Lifted Math functions
	var unaryFns = [
		'floor', 'ceil', 'round', 'sqrt', 'exp', 'log', 'abs', 'sin', 'cos',
		'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh',
		'acosh', 'atanh'
	];
	var binaryFns = [
		'pow', 'min', 'max', 'atan2'
	];
	for (var i = 0; i < unaryFns.length; i++) {
		var fnname = unaryFns[i];
		var forward = OutputType === Tensor ?
			new Function('x', 'return x.' + fnname + '();') :
			new Function('x', 'return Math.' + fnname + '(x);');
		fns[fnname] = func.newUnaryFunction(OutputType, {
			forward: forward,
			backward: backward(derivs[fnname]),
		});
	}
	for (var i = 0; i < binaryFns.length; i++) {
		var fnname = binaryFns[i];
		var forward = OutputType === Tensor ?
			new Function('x', 'y', 'return x.' + fnname + '(y);') :
			new Function('x', 'y', 'return Math.' + fnname + '(x, y);');
		fns[fnname] = func.newBinaryFunction(OutputType, {
			forward: forward,
			backward1: backward(derivs[fnname])[0],
			backward2: backward(derivs[fnname])[1]
		});
	}

	return fns;
}


var fns = {
	scalar: makeFunctions(Scalar),
	tensor: makeFunctions(Tensor)
};


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

fns.scalar.ge = func.liftBinaryFunction(
	function(x, y) { return x >= y; }
);

fns.scalar.le = func.liftBinaryFunction(
	function(x, y) { return x <= y; }
);



// Hybrid scalar/tensor operations --------------------------------------------


// Select one element out of a tensor (by linear indexing)
fns.tensorSelect = func.newFunction(Scalar, {
	forward: function(t, i) {
		return graph.isNode(t) ? t.x.data[i] : t.data[i];
	},
	backward: function(t, i) {
		if (graph.isNode(t)) {
			t.dx.data[i] += this.dx;
		}
	},
	getParents: function(t, i) {
		return graph.isNode(t) ? [t] : [];
	}
});

// Split a tensor into multiple scalars
fns.tensorSplit = function(t) {
	var n = graph.isNode(t) ? t.x.length : t.length;
	var s = new Array(n);
	while (n--) {
		s[n] = fns.tensorSelect(t, n);
	}
	return s;
}

// Concatentate multiple scalars into a tensor
// Can either take an array of scalars or a variable number of arguments
fns.scalarConcat = func.newFunction(Tensor, {
	forward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		var t = new Tensor([n]);
		while (n--) {
			var arg = args[n];
			t.data[n] = graph.isNode(arg) ? arg.x : arg;
		}
		return t;
	},
	backward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		while (n--) {
			var arg = args[n];
			if (graph.isNode(arg)) {
				arg.dx += this.dx.data[n];
			}
		}
	},
	getParents: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var p = [];
		var n = args.length;
		while (n--) {
			var arg = args[n];
			if (graph.isNode(arg)) {
				p.push(arg);
			}
		}
		return p;
	}
});

// Concatentate multiple tensors into one big tensor
// Can either take an array of tensors or a variable number of arguments
fns.tensorConcat = func.newFunction(Tensor, {
	forward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		var size = 0;
		while (n--) size += args[n].length;
		var t = new Tensor([size]);
		n = args.length;
		var i = 0;
		while (n--) {
			var arg = args[n];
			var tn = graph.isNode(arg) ? arg.x : arg;
			t.data.set(tn.data, i);
			i += tn.length;
		}
		return t;
	},
	backward: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var n = args.length;
		var i = 0;
		while (n--) {
			var arg = args[n];
			if (graph.isNode(arg)) {
				var tn = arg;
				var len = tn.dx.length;
				while (len--) {
					tn.dx.data[len] += this.dx.data[i + len];
				}
				i += tn.dx.length;
			} else i += arg.length;
		}
	},
	getParents: function() {
		var args = arguments.length === 1 && arguments[0] instanceof Array ?
			arguments[0] : arguments;
		var p = [];
		var n = args.length;
		while (n--) {
			var arg = args[n];
			if (graph.isNode(arg)) {
				p.push(arg);
			}
		}
		return p;
	}
});



module.exports = fns;




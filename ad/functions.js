var Tensor = require('../tensor.js');
var graph = require('./graph.js');
var adfn = require('./function.js');
var derivs = require('./derivatives.js');


// Make scalar & tensor operators and math functions
function makeFunctions(OutputType) {

	var fns {};

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
		fs[op] = adfn.newBinaryFunction(OutputType, {
			forward: ops[op],
			backward1: backward(derivs[op])[0],
			backward2: backward(derivs[op])[1]; }
		});
	}

	// Lifted Math functions
	fns.math = {};
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
		fns.math[fnname] = adfn.newUnaryFunction(OutputType, {
			forward: forward,
			backward: backward(derivs[fnname]),
		});
	}
	for (var i = 0; i < binaryFns.length; i++) {
		var fnname = binaryFns[i];
		var forward = OutputType === Tensor ?
			new Function('x', 'y', 'return x.' + fnname + '(y);') :
			new Function('x', 'y', 'return Math.' + fnname + '(x, y);');
		fns.math[fnname] = adfn.newBinaryFunction(OutputType, {
			forward: forward,
			backward1: backward(derivs[fnname])[0],
			backward2: backward(derivs[fnname])[1]; }
		});
	}

	return fns;
}


var fns = {
	scalar: makeFunctions(Number),
	tensor: makeFunctions(Tensor)
};


// Also lift scalar comparators

fns.scalar.eq = adfn.liftBinaryFunction(
	function(x, y) { return x == y; }
);

fns.scalar.neq = adfn.liftBinaryFunction(
	function(x, y) { return x != y; }
);

fns.scalar.peq = adfn.liftBinaryFunction(
	function(x, y) { return x === y; }
);

fns.scalar.pneq = adfn.liftBinaryFunction(
	function(x, y) { return x !== y; }
);

fns.scalar.gt = adfn.liftBinaryFunction(
	function(x, y) { return x > y; }
);

fns.scalar.lt = adfn.liftBinaryFunction(
	function(x, y) { return x < y; }
);

fns.scalar.ge = adfn.liftBinaryFunction(
	function(x, y) { return x >= y; }
);

fns.scalar.le = adfn.liftBinaryFunction(
	function(x, y) { return x <= y; }
);



module.exports = fns;




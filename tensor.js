var ArrayType = Float64Array;

function Tensor(dims) {
	this.dims = dims;
	var size = 1;
	var n = dims.length;
	while (n--) size *= dims[n];
	this.length = size;
	this.data = new ArrayType(size);
	return this;
}

Object.defineProperties(Tensor.prototype, {
	rank: { get: function() { return this.dims.length; } },
	// For interacting with 1, 2, and 3-d Tensors
	width: { get: function() { return this.dims[0]; } },
	height: { get: function() { return this.dims[1]; } },
	depth: { get: function() { return this.dims[2]; } }
});

Tensor.prototype.fill = function(val) {
	// TODO: Use TypedArray.fill, when it is more broadly supported
	var n = this.length;
	while (n--) this.data[n] = val;
	return this;
};

Tensor.prototype.zero = function() {
	return this.fill(0);
};

Tensor.prototype.clone = function() {
	var copy = new Tensor(this.dims);
	copy.data.set(this.data);
	return copy;
};


// These are slow; don't use them inside any hot loops (i.e. they're good for
//    debgugging/translating data to/from other formats, and not much else)
Tensor.prototype.get = function(coords) {
	var idx = 0;
	var n = this.dims.length;
	while (n--) {
		idx = idx * this.dims[n] + coords[n];
	}
	return this.data[idx];
};
Tensor.prototype.set = function(coords, val) {
	var idx = 0;
	var n = this.dims.length;
	while (n--) {
		idx = idx * this.dims[n] + coords[n];
	}
	this.data[idx] = val;
};
function toArrayRec(tensor, coords) {
	if (coords.length === tensor.rank) {
		return tensor.get(coords);
	} else {
		var dim = tensor.rank - coords.length - 1;
		var arr = [];
		for (var i = 0; i < tensor.dims[dim]; i++) {
			arr.push(toArrayRec(tensor, [i].concat(coords)));
		}
		return arr;
	}
}
Tensor.prototype.toArray = function() {
	return toArrayRec(this, []);
};
function fromArrayRec(tensor, coords, x) {
	if (!(x instanceof Array)) {
		tensor.set(coords, x);
	} else {
		var dim = tensor.rank - coords.length - 1;
		for (var i = 0; i < tensor.dims[dim]; i++) {
			fromArrayRec(tensor, [i].concat(coords), x[i]);
		}
	}
}
Tensor.prototype.fromArray = function(arr) {
	fromArrayRec(this, [], arr);
	return this;
};

Tensor.prototype.toString = function() {
	return this.toArray();
};



function addUnaryMethod(name, fncode) {
	var fneq = new Function([
		'var n = this.data.length;',
		'while (n--) {',
		'	var x = this.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}'
	].join('\n'));
	Tensor.prototype[name + 'eq'] = fneq;
	Tensor.prototype[name] = function() {
		var nt = this.clone();
		fneq.call(nt);
		return nt;
	};
}

function addBinaryMethod(name, fncode) {
	var fneqS = new Function('s', [
		'var n = this.data.length;',
		'var b = s;',
		'while (n--) {',
		'	var a = this.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}'
	].join('\n'));
	var fneqT = new Function('t', [
		'var n = this.data.length;',
		'while (n--) {',
		'	var a = this.data[n];',
		'	var b = t.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}'
	].join('\n'));

	var fneq = function(x) {
		if (x.constructor === Tensor)
			fneqT.call(this, x);
		else
			fneqS.call(this, x);
	}
	Tensor.prototype[name + 'eq'] = fneq;
	Tensor.prototype[name] = function(x) {
		var nt = this.clone();
		fneq.call(nt, x);
		return nt;
	};
}


addUnaryMethod('neg', '-x');
addUnaryMethod('round', 'Math.round(x)');
addUnaryMethod('log', 'Math.log(x)');
addUnaryMethod('exp', 'Math.exp(x)');
addUnaryMethod('sqrt', 'Math.sqrt(x)');
addUnaryMethod('abs', 'Math.abs(x)');
addUnaryMethod('ceil', 'Math.ceil(x)');
addUnaryMethod('floor', 'Math.floor(x)');
addUnaryMethod('cos', 'Math.cos(x)');
addUnaryMethod('sin', 'Math.sin(x)');
addUnaryMethod('tan', 'Math.tan(x)');
addUnaryMethod('acos', 'Math.acos(x)');
addUnaryMethod('asin', 'Math.asin(x)');
addUnaryMethod('atan', 'Math.atan(x)');
addUnaryMethod('cosh', 'Math.cosh(x)');
addUnaryMethod('sinh', 'Math.sinh(x)');
addUnaryMethod('tanh', 'Math.tanh(x)');
addUnaryMethod('acosh', 'Math.acosh(x)');
addUnaryMethod('asinh', 'Math.asinh(x)');
addUnaryMethod('atanh', 'Math.atanh(x)');

addBinaryMethod('add', 'a + b');
addBinaryMethod('sub', 'a - b');
addBinaryMethod('mul', 'a * b');
addBinaryMethod('div', 'a / b');
addBinaryMethod('mod', 'a % b');
addBinaryMethod('min', 'Math.min(a, b)');
addBinaryMethod('max', 'Math.max(a, b)');
addBinaryMethod('pow', 'Math.pow(a, b)');
addBinaryMethod('atan2', 'Math.atan2(a, b)');



module.exports = Tensor;




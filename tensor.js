var assert = require('assert');
var utils = require('./utils.js');


// Can swap out different backing stores
function TypedArrayBackingStore(ArrayType) {
	return {
		new: function(n) { return new ArrayType(n); },
		set: function(tgt, src, offset) {
			tgt.set(src, offset);
		}
	}
}
var ArrayBackingStore = {
	ArrayType: Array,
	new: function(n) {
		var a = new Array(n);
		while (n--) { a[n] = 0; }
		return a;
	},
	set: function(tgt, src, offset) {;
		for (var i = 0; i < src.length; i++) {
			tgt[i+offset] = src[i];
		}
	}
};


// The actual backing store we're using
var BackingStore = TypedArrayBackingStore(Float64Array);


function Tensor(dims) {
	this.dims = dims;
	var size = 1;
	var n = dims.length;
	while (n--) size *= dims[n];
	this.length = size;
	this.data = BackingStore.new(size);
	return this;
}

Object.defineProperties(Tensor.prototype, {
	rank: { get: function() { return this.dims.length; } },
});

Tensor.prototype.reshape = function(dims) {
	var size = 1;
	var n = dims.length;
	while (n--) size *= dims[n];
	assert(size === this.length, 'Tensor reshape invalid size');
	this.dims = dims;
}

Tensor.prototype.fill = function(val) {
	// TODO: Use TypedArray.fill, when it is more broadly supported
	var n = this.length;
	while (n--) this.data[n] = val;
	return this;
};

Tensor.prototype.zero = function() {
	return this.fill(0);
};

// Adapted from:
//    https://github.com/karpathy/convnetjs/blob/master/src/convnet_vol.js
Tensor.prototype.fillRandom = function() {
	var scale = 1/this.length;
	var n = this.length;
	while (n--) this.data[n] = utils.gaussianSample(0, scale);
	return this;
}

Tensor.prototype.copy = function(other, offset) {
	offset = offset || 0;
	BackingStore.set(this.data, other.data, offset);
	return this;
};

Tensor.prototype.clone = function() {
	var copy = new Tensor(this.dims);
	return copy.copy(this);
};


// These are slow; don't use them inside any hot loops (i.e. they're good for
//    debgugging/translating data to/from other formats, and not much else)
Tensor.prototype.get = function(coords) {
	var idx = 0;
	var n = this.dims.length;
	for (var i = 0; i < n; i++) {
		idx = idx * this.dims[i] + coords[i];
	}
	return this.data[idx];
};
Tensor.prototype.set = function(coords, val) {
	var idx = 0;
	var n = this.dims.length;
	for (var i = 0; i < n; i++) {
		idx = idx * this.dims[i] + coords[i];
	}
	this.data[idx] = val;
};
function toArrayRec(tensor, coords) {
	if (coords.length === tensor.rank) {
		return tensor.get(coords);
	} else {
		var dim = coords.length;
		var arr = [];
		for (var i = 0; i < tensor.dims[dim]; i++) {
			arr.push(toArrayRec(tensor, coords.concat([i])));
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
		var dim = coords.length;
		for (var i = 0; i < tensor.dims[dim]; i++) {
			fromArrayRec(tensor, coords.concat([i]), x[i]);
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


Tensor.prototype.toFlatArray = function() {
	return Array.prototype.slice.call(this.data);
}
Tensor.prototype.fromFlatArray = function(arr) {
	BackingStore.set(this.data, arr, 0);
	return this;
}



function addUnaryMethod(name, fncode) {
	var fneq = new Function([
		'var n = this.data.length;',
		'while (n--) {',
		'	var x = this.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}',
		'return this;'
	].join('\n'));
	Tensor.prototype[name + 'eq'] = fneq;
	Tensor.prototype[name] = function() {
		var nt = this.clone();
		return fneq.call(nt);
	};
}

function addBinaryMethod(name, fncode) {
	var fneqS = new Function('s', [
		'var n = this.data.length;',
		'var b = s;',
		'while (n--) {',
		'	var a = this.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}',
		'return this;'
	].join('\n'));
	var fneqT = new Function('t', [
		'var n = this.data.length;',
		'while (n--) {',
		'	var a = this.data[n];',
		'	var b = t.data[n];',
		'	this.data[n] = ' + fncode + ';',
		'}',
		'return this;'
	].join('\n'));

	var fneq = function(x) {
		if (x.constructor === Tensor)
			return fneqT.call(this, x);
		else
			return fneqS.call(this, x);
	}
	Tensor.prototype[name + 'eq'] = fneq;
	Tensor.prototype[name] = function(x) {
		var nt = this.clone();
		return fneq.call(nt, x);
	};
}

function addReduction(name, initcode, fncode) {
	Tensor.prototype[name+'reduce'] = new Function([
		'var accum = ' + initcode + ';',
		'var n = this.data.length;',
		'while (n--) {',
		'	var x = this.data[n];',
		'	accum = ' + fncode + ';',
		'}',
		'return accum;'
	].join('\n'));
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
addUnaryMethod('sigmoid', '1 / (1 + Math.exp(-x))');
addUnaryMethod('isFinite', 'isFinite(x)');
addUnaryMethod('isNaN', 'isNaN(x)');
addUnaryMethod('invert', '1/x');
addUnaryMethod('pseudoinvert', 'x === 0 ? 0 : 1/x');

addBinaryMethod('add', 'a + b');
addBinaryMethod('sub', 'a - b');
addBinaryMethod('mul', 'a * b');
addBinaryMethod('div', 'a / b');
addBinaryMethod('mod', 'a % b');
addBinaryMethod('min', 'Math.min(a, b)');
addBinaryMethod('max', 'Math.max(a, b)');
addBinaryMethod('pow', 'Math.pow(a, b)');
addBinaryMethod('atan2', 'Math.atan2(a, b)');
addBinaryMethod('eq', 'a === b');
addBinaryMethod('neq', 'a !== b');
addBinaryMethod('gt', 'a > b');
addBinaryMethod('ge', 'a >= b');
addBinaryMethod('lt', 'a < b');
addBinaryMethod('le', 'a <= b');

addReduction('sum', '0', 'accum + x');
addReduction('min', 'Infinity', 'Math.min(accum, x)');
addReduction('max', '-Infinity', 'Math.max(accum, x)');
addReduction('all', 'true', 'accum && (x !== 0)');
addReduction('any', 'false', 'accum || (x !== 0)');


module.exports = Tensor;




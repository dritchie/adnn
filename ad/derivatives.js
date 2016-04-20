'use strict';

var nop = function() {};

// Make backwards pass derivative functions for both scalar and tensor
//    operations using the same source code.

function makeUnaryDerivatives(code) {
	if (code === undefined) {
		return { scalar: nop, tensor: nop };
	} else {
		return {
			scalar: new Function('_x', [
				'var x = _x.x;',
				'var out = this.x;',
				'_x.dx += (' + code + ') * this.dx;'
			].join('\n')),
			tensor: new Function('_x', [
				'var n = _x.x.length;',
				'while (n--) {',
				'	var x = _x.x.data[n];',
				'	var out = this.x.data[n];',
				'   _x.dx.data[n] += (' + code + ') * this.dx.data[n];',
				'}'
			].join('\n'))
		};
	}
}

function makeBinaryDerivatives(code1, code2) {
	if (code1 === undefined && code2 === undefined) {
		return { scalar: [nop, nop], tensor: [nop, nop] };
	} else  {
		return {
			scalar: [
				// First arg is definitely a Node, second may or may not be
				new Function('_x', '_y', [
					'var x = _x.x;',
					'var y = (typeof _y === "number") ? _y : _y.x;',
					'var out = this.x;',
					'_x.dx += (' + code1 + ') * this.dx;'
				].join('\n')),
				// Second arg is definitely a Node, first may or may not be
				new Function('_x', '_y', [
					'var x = (typeof _x === "number") ? _x : _x.x;',
					'var y = _y.x;',
					'var out = this.x;',
					'_y.dx += (' + code2 + ') * this.dx;'
				].join('\n'))
			],
			// To match the implementations of the methods on Tensor objects,
			//    the second argument might be a scalar or a Tensor.
			tensor: [
				// First arg is definitely a Node, second may or may not be
				new Function('_x', '_y', [
					'var _xx = _x.x;',
					'var _yx = _y.x || _y;',
					'var n = _xx.length;',
					// y is a scalar
					'if (typeof _yx === "number") {',
					'	while (n--) {',
					'		var x = _xx.data[n];',
					'		var y = _yx;',
					'		var out = this.x.data[n];',
					'	   _x.dx.data[n] += (' + code1 + ') * this.dx.data[n];',
					'	}',
					// y is a tensor 
					'} else {',
					'	while (n--) {',
					'		var x = _xx.data[n];',
					'		var y = _yx.data[n];',
					'		var out = this.x.data[n];',
					'	   _x.dx.data[n] += (' + code1 + ') * this.dx.data[n];',
					'	}',
					'}',
				].join('\n')),
				// Second arg is definitely a Node, first may or may not be
				new Function('_x', '_y', [
					'var _xx = _x.x || _x;',
					'var _yx = _y.x;',
					'var n = _xx.length;',
					// y is a scalar
					'if (typeof _yx === "number") {',
					'	while (n--) {',
					'		var x = _xx.data[n];',
					'		var y = _yx;',
					'		var out = this.x.data[n];',
					'	   _y.dx += (' + code2 + ') * this.dx.data[n];',
					'	}',
					// y is a tensor
					'} else {',
					'	while (n--) {',
					'		var x = _xx.data[n];',
					'		var y = _yx.data[n];',
					'		var out = this.x.data[n];',
					'	   _y.dx.data[n] += (' + code2 + ') * this.dx.data[n];',
					'	}',
					'}'
				].join('\n'))
			]
		};
	}
}


var d = {};

d.neg = makeUnaryDerivatives('-1');
d.add = makeBinaryDerivatives('1', '1');
d.sub = makeBinaryDerivatives('1', '-1');
d.mul = makeBinaryDerivatives('y', 'x');
d.div = makeBinaryDerivatives('1/y', '-x/(y*y)');
d.sqrt = makeUnaryDerivatives('1/(2*out)');
d.exp = makeUnaryDerivatives('out');
d.log = makeUnaryDerivatives('1/x');
d.pow = makeBinaryDerivatives('y*Math.pow(x,y-1)', 'Math.log(x)*out');
d.sin = makeUnaryDerivatives('Math.cos(x)');
d.cos = makeUnaryDerivatives('-Math.sin(x)');
d.tan = makeUnaryDerivatives('1 + out*out');
d.asin = makeUnaryDerivatives('1 / Math.sqrt(1 - x*x)');
d.acos = makeUnaryDerivatives('-1 / Math.sqrt(1 - x*x)');
d.atan = makeUnaryDerivatives('1 / (1 + x*x)');
d.atan2 = makeBinaryDerivatives('y/(x*x + y*y)', '-x/(x*x + y*y)');
d.sinh = makeUnaryDerivatives('Math.cosh(x)');
d.cosh = makeUnaryDerivatives('Math.sinh(x)');
d.tanh = makeUnaryDerivatives('1 - out*out');
d.asinh = makeUnaryDerivatives('1 / Math.sqrt(x*x + 1)');
d.acosh = makeUnaryDerivatives('1 / Math.sqrt(x*x - 1)');
d.atanh = makeUnaryDerivatives('1 / (1 - x*x)');
d.sigmoid = makeUnaryDerivatives('out * (1 - out)');

// Functions with no derivative
d.floor = makeUnaryDerivatives();
d.ceil = makeUnaryDerivatives();
d.round = makeUnaryDerivatives();
d.abs = makeUnaryDerivatives();
d.min = makeBinaryDerivatives();
d.max = makeBinaryDerivatives();


module.exports = d;







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
				'var n = _x.length;',
				'while (n--) {',
				'	var x = _x.x[n];',
				'	var out = this.x[n];',
				'   _x.dx[n] += (' + code + ') * this.dx[n];',
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
				new Function('_x', '_y', [
					'var x = _x.x;',
					'var y = _y.x;',
					'var out = this.x;',
					'_x.dx += (' + code1 + ') * this.dx;'
				].join('\n')),
				new Function('_x', '_y', [
					'var x = _x.x;',
					'var y = _y.x;',
					'var out = this.x;',
					'_y.dx += (' + code2 + ') * this.dx;'
				].join('\n'))
			],
			tensor: [
				new Function('_x', '_y', [
					'var n = _x.length;',
					'while (n--) {',
					'	var x = _x.x[n];',
					'	var y = _y.x[n];',
					'	var out = this.x[n];',
					'   _x.dx[n] += (' + code1 + ') * this.dx[n];',
					'}'
				].join('\n')),
				new Function('_x', '_y', [
					'var n = _x.length;',
					'while (n--) {',
					'	var x = _x.x[n];',
					'	var y = _y.x[n];',
					'	var out = this.x[n];',
					'   _y.dx[n] += (' + code2 + ') * this.dx[n];',
					'}'
				].join('\n'))
			]
		};
	}
}


var d = {};

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

// Functions with no derivative
d.floor = makeUnaryDerivatives();
d.ceil = makeUnaryDerivatives();
d.round = makeUnaryDerivatives();
d.abs = makeUnaryDerivatives();
d.min = makeUnaryDerivatives();
d.max = makeUnaryDerivatives();


module.exports = d;






var ffith = require('/Users/jpchen/jstorch/torch.js/TH.js')
var TH = ffith.TH

'use strict';

var nop = function() {};

// Make backwards pass derivative functions for both scalar and tensor
//    operations using the same source code.
// TH C functions added in libTH
var d = {};

function makeUnaryDerivatives(key, code, arr_args) {
    if (code === undefined) {
        d[key] = { scalar: nop, tensor: nop };
    } else {
        var fct_name = "dx_" + key;
        var arr_args = []
        if(code.indexOf("x") != -1)
            arr_args.push('_x.x.data.ref()')
        if(code.indexOf("out") != -1)
            arr_args.push('this.x.data.ref()')

        d[key] = {
            scalar: new Function('_x', [
                'var x = _x.x;',
                'var out = this.x;',
                '_x.dx += (' + code + ') * this.dx;'
            ].join('\n')),
            tensor: new Function('_x', [
                //'Use added torch dx functions
                'var TH = _x.x.ffi;',
                'TH.THFloatTensor_' + fct_name + '(_x.dx.data.ref(), this.dx.data.ref()' + (arr_args.length > 0 ? ", " + arr_args.join(",") : "") + ')'
            ].join('\n'))
        };
    }
}

function makeBinaryDerivatives(key, code1, code2) {
    if (code1 === undefined && code2 === undefined) {
        d[key] =  { scalar: [nop, nop], tensor: [nop, nop] };
    } else  {

        var make_arg_array = function(cc, y_is_full) {
            var fct_args = []
            if(cc.indexOf("x") != -1)
                fct_args.push('_x.x.data.ref()')
            if(cc.indexOf("out") != -1)
                fct_args.push('this.x.data.ref()')
            
            // if constant exists, it's always the last reference sent in
            if(cc.indexOf("y") != -1)
                fct_args.push(y_is_full ? '_y.x.data' : '_yx')
            return fct_args
        }

        var fct_name = "dx_" + key;
        var fct_acc_name = "dx_acc_" + key;
        var fct_const_name = "dx_const_" + key;
        
        
        // if our y is a constant, these are what get sent into the function
        var t1_full_args = make_arg_array(code1, true)
        var t1_const_args = make_arg_array(code1, false)
        // set our function name for f1s
        var t1_const_fct_name = fct_const_name + "_f1"
        var t1_fct_name = fct_name + "_f1"
        // if we have no arguments, then we're a 0 arg sum thing
        // just use the normal function name dx_
        if(t1_const_args.length == 0)
            t1_const_fct_name = fct_name + "_f1"


        var t2_full_args = make_arg_array(code2, true)
        var t2_acc_args = make_arg_array(code2, false)
        var t2_acc_fct_name = fct_acc_name + "_f2";
        var t2_fct_name = fct_name + "_f2"

        
        d[key] = {
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
                    'var TH = _x.x.ffi',
                    //'console.log("'+key+'")',
                    // y is a scalar
                    'if (typeof _yx === "number") {',
                    'TH.THFloatTensor_' + t1_const_fct_name + '(_x.dx.data.ref(), this.dx.data.ref()' + (t1_const_args.length > 0 ? ", " + t1_const_args.join(",") : "") + ')',
                    // y is a tensor 
                    '} else {',
                    'TH.THFloatTensor_' + t1_fct_name + '(_x.dx.data.ref(), this.dx.data.ref()' + (t1_full_args.length > 0 ? ", " + t1_full_args.join(",") : "") + ')',
                    '}',
                ].join('\n')),
                // Second arg is definitely a Node, first may or may not be
                new Function('_x', '_y', [
                    'var _xx = _x.x || _x;',
                    'var _yx = _y.x;',
                    'var n = _xx.length;',
                    'var TH = _x.x.ffi',
                    // y is a scalar
                    'if (typeof _yx === "number") {',
                    '_y.dx += TH.THFloatTensor_' + t2_acc_fct_name + '(this.dx.data.ref()' + (t2_acc_args.length > 0 ? ", " + t2_acc_args.join(",") : "") + ')',
                    '} else {',
                    'TH.THFloatTensor_' + t2_fct_name + '(_y.dx.data.ref(), this.dx.data.ref()' + (t2_full_args.length > 0 ? ", " + t2_full_args.join(",") : "") + ')',
                    '}'
                ].join('\n'))
            ]
        };
    }
}

// tell it about the equivalent tensor call
makeUnaryDerivatives('neg', '-1');

// these will auto-check the required arguments for sending to C
makeUnaryDerivatives('sqrt', '1/(2*out)');
makeUnaryDerivatives('exp', 'out');
makeUnaryDerivatives('log', '1/x');

makeUnaryDerivatives('sin', 'Math.cos(x)');
makeUnaryDerivatives('cos', '-Math.sin(x)');
makeUnaryDerivatives('tan', '1 + out*out');
makeUnaryDerivatives('asin', '1 / Math.sqrt(1 - x*x)');
makeUnaryDerivatives('acos', '-1 / Math.sqrt(1 - x*x)');
makeUnaryDerivatives('atan', '1 / (1 + x*x)');
makeUnaryDerivatives('abs', 'x >= 0 ? 1 : -1');

makeUnaryDerivatives('sinh', 'Math.cosh(x)');
makeUnaryDerivatives('cosh', 'Math.sinh(x)');
makeUnaryDerivatives('tanh', '1 - out*out');
makeUnaryDerivatives('asinh', '1 / Math.sqrt(x*x + 1)');
makeUnaryDerivatives('acosh', '1 / Math.sqrt(x*x - 1)');
makeUnaryDerivatives('atanh', '1 / (1 - x*x)');
makeUnaryDerivatives('sigmoid', 'out * (1 - out)');

makeBinaryDerivatives('add', '1', '1');
makeBinaryDerivatives('sub', '1', '-1');
makeBinaryDerivatives('mul', 'y', 'x');
makeBinaryDerivatives('div', '1/y', '-x/(y*y)');
makeBinaryDerivatives('pow', 'y*Math.pow(x,y-1)', 'Math.log(x)*out');
makeBinaryDerivatives('atan2', 'y/(x*x + y*y)', '-x/(x*x + y*y)');

// Functions with no derivative
makeUnaryDerivatives('floor');
makeUnaryDerivatives('ceil');
makeUnaryDerivatives('round');

makeBinaryDerivatives('min');
makeBinaryDerivatives('max');

module.exports = d;


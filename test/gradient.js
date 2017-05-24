var assert = require('assert');
var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');
var _ = require('lodash');
var T = ad.tensor

var _t1 = [[1, 3, 5],
           [2, 4, 6],
           [9, 7, 3]];
var eps = 1e-8;

//Finite differences
// Returns true if the finite difference between computed
// and actual is 0, up to 4 sig figs
function partial_deriv(x, fn, scal, type, verbose) {
    _x = x.x;
    var n = _x.length;
    result =  [];
    if (type === "Torch") {
        for (var i = 0; i < _x.dims[0]; i++) {
            for (var j = 0; j < _x.dims[1]; j++) {
                var epsT = new THTensor(_x.dims).zero();
                epsT.set([i,j], eps);
//                 console.log(T.add(_x,epsT).toArray())
                var y1 = scal ? f_1(_x, fn) : f(_x, fn);
                var y2 = scal ? f_1(T.add(_x, epsT), fn)
                    : f(T.add(_x, epsT), fn);
                console.log(y1, y2)
                if (verbose) {
                    console.log('finite diffs: ', (y2-y1) / eps);
                    console.log('ad: ', T.get(ad.derivative(x), [i, j]));
                }
                result.push((y2-y1) / eps -  T.get(ad.derivative(x), [i, j])) ;
            }
        }
    } else {
        for (var i = 0; i < n; i++) {
            var epsarr = Array(n).fill(0);
            epsarr[i] = 1e-8;
            var y1 = scal ? f_1(_x, fn) : f(_x, fn);
            var y2 = scal ? f_1(T.add(_x, new Tensor(_x.dims).fromFlatArray(epsarr)), fn)
                : f(T.add(_x, new Tensor(_x.dims).fromFlatArray(epsarr)), fn);
            if (verbose) {
                console.log('finite diffs: ', (y2-y1) / eps);
                console.log('ad: ', T.get(ad.derivative(x), i));
            }
            result.push((y2-y1) / eps -  T.get(ad.derivative(x), i)) ;
        }
    }
    return _.reduce(roundArr(result, 4), function(sum, n) {
        return sum + n;
    }, 0);
}

function bin_partial_deriv(x, y, fn, type, verbose) {
    _x = x.x;
    _y = y.x;
    var n = _x.length;
    result =  [];
    if (type === "Torch") {
        for (var i = 0; i < _x.dims[0]; i++) {
            for (var j = 0; j < _x.dims[1]; j++) {
                var epsT = new THTensor(_x.dims).zero();
                epsT.set([i,j], eps);
                var y1 = f_2(_x, _y, fn);
                var y2 = f_2(T.add(_x, epsT), _y, fn);
                if (verbose) {
                    console.log('_x finite diffs: ', (y2-y1) / eps);
                    console.log('ad: ', T.get(ad.derivative(x), [i, j]));
                }
                result.push((y2-y1) / eps -  T.get(ad.derivative(x), [i, j])) ;
            }
        }
        for (var i = 0; i < _x.dims[0]; i++) {
            for (var j = 0; j < _x.dims[1]; j++) {
                var epsT = new THTensor(_x.dims).zero();
                epsT.set([i,j], eps);
                var y1 = f_2(_x, _y, fn);
                var y2 = f_2(_x, T.add(_y, epsT), fn);
                if (verbose) {
                    console.log('_x finite diffs: ', (y2-y1) / eps);
                    console.log('ad: ', T.get(ad.derivative(x), [i,j]));
                }
                result.push((y2-y1) / eps -  T.get(ad.derivative(x), [i, j])) ;
            }
        }
    } else {
        for (var i = 0; i < n; i++) {
            var epsarr = Array(n).fill(0);
            epsarr[i] = 1e-8;
            var y1 = f_2(_x, _y, fn);
            var y2 = f_2(T.add(_x, new Tensor(_x.dims).fromFlatArray(epsarr)), _y, fn);
            if (verbose) {
                console.log('_x finite diffs: ', (y2-y1) / eps);
                console.log('ad: ', T.get(ad.derivative(x), i));
            }
            result.push((y2-y1) / eps -  T.get(ad.derivative(x), i)) ;
        }
        for (var i = 0; i < n; i++) {
            var epsarr = Array(n).fill(0);
            epsarr[i] = 1e-8;
            var y1 = f_2(_x, _y, fn);
            var y2 = f_2(_x, T.add(_y, new Tensor(_y.dims).fromFlatArray(epsarr)), fn);
            if (verbose) {
                console.log('_y finite diffs: ', (y2-y1) / eps);
                console.log('ad: ', T.get(ad.derivative(y), i));
            }
            result.push((y2-y1) / eps -  T.get(ad.derivative(y), i)) ;
        }
    }
    return _.reduce(roundArr(result, 4), function(sum, n) {
        return sum + n;
    }, 0);
}

var f = function(x, fn) {
  var out = T.sumreduce(fn(x));
  return out;
};

var f_1 = function(x, fn) {
  var out = fn(x);
  return out;
};

var f_2 = function(x, y, fn) {
  var out = T.sumreduce(fn(x, y));
  return out;
}

var bp = function(_x, op, type, scal) {
  var x = op === 'concat' ? _x.map(function(v) {return ad.lift(v);}) 
      : ad.lift(_x);
  var xnode = scal ? xnode = f_1(x, function(n) {return T[op](n); })
      : xnode = f(x, function(n) {return T[op](n); });
  xnode.backprop();
  return partial_deriv(x, function(n) {return T[op](n);}, scal, type, true);
}

var bin_bp = function(_x, _y, op, type) {
  var x = ad.lift(_x);
  var y = ad.lift(_y);
  var onode = f_2(x, y, function(n, m) {return T[op](n, m); });
  onode.backprop();
  return bin_partial_deriv(x, y, function(n, m) {return T[op](n, m)}, type);
}

function roundArr (arr, sf) {
  var i = 0;
  while(i < arr.length){ 
    if (arr[i] instanceof Array) {
        for (var j = 0; j < arr[i].length; j++) {
            arr[i][j] = parseFloat(arr[i][j].toFixed(sf));
        }
    } else {
        arr[i] = parseFloat(arr[i].toFixed(sf)); 
    }
    i++;
  }
  return arr;
}

function fillArr (el, len) {
  var arr = []
  for (var i = 0; i < len; i++) {
    arr.push(el);
  }
  return arr;
}

function run (type) {
    var tensortype = type === "Torch" ? "thtensor" : "tensor";
    var x = tensortype === "tensor" ? new Tensor([3,3]).fromArray(_t1) : new THTensor([3,3]).fromArray(_t1);
    var y = tensortype === "tensor" ? new Tensor([3,3]).fill(2) : new THTensor([3,3]).fill(2);
    var x1d = tensortype === "tensor" ? new Tensor([3]).fill(2) : new THTensor([3]).fill(2);
    describe(type, function () {
        describe("Math Ops", function () {
            describe("Unary", function () {
                //Unary operators
                it('tanh', function () { assert.equal(bp(x, 'tanh', type), 0); }); 
                it('neg', function () { assert.equal(bp(x, 'neg', type), 0); }); 
                it('sqrt', function () { assert.equal(bp(x, 'sqrt', type), 0); }); 
                it('exp', function () { assert.equal(bp(x, 'exp', type).toFixed(2), 0); }); 
                it('log', function () { assert.equal(bp(x, 'log', type), 0); }); 
                //no abs derivative for TH yet
//                 it('abs', function () { assert.equal(bp(x, 'abs', type), 0); }); 
                it('sin', function () { assert.equal(bp(x, 'sin', type), 0); }); 
                it('cos', function () { assert.equal(bp(x, 'cos', type), 0); }); 
                it('tan', function () { assert.equal(bp(x, 'tan', type), 0); }); 
                it('sigmoid', function () { assert.equal(bp(x, 'sigmoid', type), 0); }); 
            });
            describe ("Binary", function () {
                //Binary operators
                it('add', function () { assert.equal(bin_bp(x, y, 'add', type), 0); }); 
                it('sub', function () { assert.equal(bin_bp(x, y, 'sub', type), 0); }); 
//                 it('mul', function () { assert.equal(bin_bp(x, y, 'mul', type), 0); }); 
//                 it('div', function () { assert.equal(bin_bp(x, y, 'div', type), 0); }); 
//                 it('pow', function () { assert.equal(bin_bp(x, y, 'pow', type), 0); }); 
//                 it('atan2', function () { assert.equal(bin_bp(x, y, 'atan2', type), 0); }); 
            });
        });
        describe("Tensor Ops", function () {
            it('sumreduce', function () { assert.equal(bp(x, 'sumreduce', type, true), 0); }); 
            //need to write tensor ops to pass proper arguments to get and range below since they take scalar args
//             it('get', function () { assert.equal(bp(x, 'get', type, true), 0); }); 
//             it('range', function () { assert.equal(bp(x1d, 'range', type), 0); }); 
//          // tested in tensorgrad
//             it('concat', function () { assert.equal(bpt([x1d, x1d], 'concat', type), 0); }); 
            it('softmax', function () { assert.equal(bp(x, 'softmax', type), 0); }); 
        });
        describe("Linear Alg Ops", function () {
            it('transpose', function () { assert.equal(bp(x, 'transpose', type), 0); }); 
            it('diagonal', function () { assert.equal(bp(x1d, 'diagonal', type), 0); }); 
//             it('diag', function () { assert.equal(bp(x, 'diag', type), 0); }); 
            it('inverse', function () { assert.equal(bp(x, 'inverse', type), 0); }); 
            it('determinant', function () { assert.equal(bp(x, 'determinant', type, true).toFixed(3), 0); }); 
            it('dot', function () { assert.equal(bin_bp(x, y, 'dot', type), 0); }); 
        });
    });
}

run ("Torch");
// run ("JS");
return

var t_in = ad.lift(d_1);
console.log(ad.tensor.relu(t_in).x.toArray());
//   return

//===========

var scalarIn = ad.lift(1.5);
var tensorIn = tensortype ==="thtensor" ? ad.lift(new THTensor([3]).fill(1.5)) : ad.lift(new Tensor([3]).fill(1.5));
var tensor2 = tensortype ==="thtensor" ? ad.lift(new THTensor([3]).fill(2)) : ad.lift(new Tensor([3]).fill(2));
//   console.log(tensorIn)

// Feeding these nodes into AD functions results in Node outputs, which can be used to
//    initialize backpropagation
var scalarOut = ad.scalar.tanh(1.5);
var tensorOut = ad.tensor.mul(tensorIn,tensor2);
console.log(tensor2.dx.toArray());
tensorOut.backprop();
console.log(tensor2.dx.toArray());
return

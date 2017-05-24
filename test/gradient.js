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
function partial_deriv(x, fn, scal, type, tol, verbose) {
    var sf = 4;
    var _x = x.x;
    var n = _x.length;
    result =  [];
    this.tol = tol ? tol : 0.1
    if (type === "Torch") {
        sf = 1;
        _x = new Tensor(x.x.dims).fromArray(x.x.toArray());
        for (var i = 0; i < _x.dims[0]; i++) {
            for (var j = 0; j < _x.dims[1]; j++) {
                var epsT = new Tensor(_x.dims).zero();
                epsT.set([i,j], eps);
                var y1 = scal ? f_1(_x, fn) : f(_x, fn);
                var y2 = scal ? f_1(T.add(_x, epsT), fn)
                    : f(T.add(_x, epsT), fn);
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
    return _.every(roundArr(result, sf), function(n) {
        return n >= -this.tol && n <= this.tol;
    }.bind(this));
}

function bin_partial_deriv(x, y, fn, type, tol, verbose) {
    var _x = x.x;
    var _y = y.x;
    var sf = 4;
    var n = _x.length;
    this.tol = tol ? tol : 0.1
    result =  [];
    if (type === "Torch") {
        _x = new Tensor(x.x.dims).fromArray(x.x.toArray());
        _y = new Tensor(y.x.dims).fromArray(y.x.toArray());
        sf = 1;
        for (var i = 0; i < _x.dims[0]; i++) {
            for (var j = 0; j < _x.dims[1]; j++) {
                var epsT = new Tensor(_x.dims).zero();
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
                var epsT = new Tensor(_x.dims).zero();
                epsT.set([i,j], eps);
                var y1 = f_2(_x, _y, fn);
                var y2 = f_2(_x, T.add(_y, epsT), fn);
                if (verbose) {
                    console.log('_x finite diffs: ', (y2-y1) / eps);
                    console.log('ad: ', T.get(ad.derivative(y), [i,j]));
                }
                result.push((y2-y1) / eps -  T.get(ad.derivative(y), [i, j])) ;
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
    return _.every(roundArr(result, sf), function(n) {
        return n >= -this.tol && n <= this.tol;
    }.bind(this));
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

var bp = function(_x, op, type, scal, tol) {
  var x = ad.lift(_x);
  var xnode = scal ? xnode = f_1(x, function(n) {return T[op](n); })
      : xnode = f(x, function(n) {return T[op](n); });
  xnode.backprop();
  return partial_deriv(x, function(n) {return T[op](n);}, scal, type, tol);
}

var bin_bp = function(_x, _y, op, type, tol) {
  var x = ad.lift(_x);
  var y = ad.lift(_y);
  var onode = f_2(x, y, function(n, m) {return T[op](n, m); });
  onode.backprop();
  return bin_partial_deriv(x, y, function(n, m) {return T[op](n, m)}, type, tol);
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
                it('tanh', function () { assert(bp(x, 'tanh', type)); }); 
                it('neg', function () { assert(bp(x, 'neg', type)); }); 
                it('sqrt', function () { assert(bp(x, 'sqrt', type)); }); 
                it('exp', function () { assert(bp(x, 'exp', type)); }); 
                it('log', function () { assert(bp(x, 'log', type)); }); 
                //no abs derivative for TH yet
//                 it('abs', function () { assert(bp(x, 'abs', type)); }); 
                it('sin', function () { assert(bp(x, 'sin', type)); }); 
                it('cos', function () { assert(bp(x, 'cos', type)); }); 
                it('tan', function () { assert(bp(x, 'tan', type)); }); 
                it('sigmoid', function () { assert(bp(x, 'sigmoid', type)); }); 
            });
            describe ("Binary", function () {
                //Binary operators
                it('add', function () { assert(bin_bp(x, y, 'add', type)); }); 
                it('sub', function () { assert(bin_bp(x, y, 'sub', type)); }); 
                it('mul', function () { assert(bin_bp(x, y, 'mul', type)); }); 
                it('div', function () { assert(bin_bp(x, y, 'div', type)); }); 
                it('pow', function () { assert(bin_bp(x, y, 'pow', type)); }); 
                it('atan2', function () { assert(bin_bp(x, y, 'atan2', type)); }); 
            });
        });
        describe("Tensor Ops", function () {
            it('sumreduce', function () { assert(bp(x, 'sumreduce', type, true)); }); 
            //need to write tensor ops to pass proper arguments to get and range below since they take scalar args
//             it('get', function () { assert(bp(x, 'get', type, true)); }); 
//             it('range', function () { assert(bp(x1d, 'range', type)); }); 
//          // tested in tensorgrad
//             it('concat', function () { assert(bpt([x1d, x1d], 'concat', type)); }); 
            it('softmax', function () { assert(bp(x, 'softmax', type, false, 0.9)); }); 
        });
        describe("Linear Alg Ops", function () {
            it('transpose', function () { assert(bp(x, 'transpose', type)); }); 
            it('diagonal', function () { assert(bp(x1d, 'diagonal', type)); }); 
            it('diag', function () { assert(bp(x, 'diag', type)); }); 
            it('inverse', function () { assert(bp(x, 'inverse', type)); }); 
            it('determinant', function () { assert(bp(x, 'determinant', type, true)); }); 
            it('dot', function () { assert(bin_bp(x, y, 'dot', type)); }); 
        });
    });
}

run ("Torch");
run ("JS");
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

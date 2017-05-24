var ad = require('../ad');
var _ = require('lodash')
var Tensor = require('../tensor');
var Tensor = require('../THTensor');
var T = ad.tensor;

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

// test back prop through cholesky
var a, b, c
var f = function(x,y) {
    a = T.sumreduce(T.concat(x));
//     a = T.log(x);
//     return a
    return a;
    c = T.mul(x,x);
    b = T.sumreduce(c);
    return ad.scalar.add(a, b);
};

var g = function(x,i) {
  a = T.get(x, i);
  return a;
}

var h = function(x,s, e) {
  console.log(T.range(x,s,e).x.toArray())
  a = T.sumreduce(T.range(x, s, e));
  return a;
}
// CONCAT
var _x = new Tensor([4]).fromArray([1,-0.5,-0.5,0.8]);
var _k = new Tensor([4]).fromArray([1,-2,-3,4]);
// var _x = new Tensor([2,2]).fromArray([[1,2],[4,4]]);
var x1 = ad.lift(_x);
var k1 = ad.lift(_k);
x = [x1, k1]
var y = f(x);
y.backprop();
//console.log(y.dx.toArray());
//console.log(ad.derivative(x).toArray())
console.log(x[0].x.toArray())
console.log(x[1].x.toArray())
// console.log(ad.derivative(a))
//
//GET
var x2 = ad.lift(_x);
y = g(x2, 1);
y.backprop();
console.log(ad.derivative(x2).toArray());
// console.log(ad.derivative(a));

//RANGE
var x3 = ad.lift(_k)
y = h(x3, 0, 3);
y.backprop();
console.log(ad.derivative(x3).toArray());
// console.log(ad.derivative(a));

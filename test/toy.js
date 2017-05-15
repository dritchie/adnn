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
    a = T.sumreduce(T.log(x));
//     a = T.log(x);
//     return a
    c = T.mul(x,x);
    b = T.sumreduce(c);
    return ad.scalar.add(a, b);
};
//
// var _x = new Tensor([2,2]).fromFlatArray([1,-0.5,-0.5,0.8]);
var _x = new Tensor([2,2]).fromArray([[1,2],[4,4]]);
var x = ad.lift(_x);
var y = f(x);
y.backprop();
// console.log(y.dx.toArray());
console.log(ad.derivative(x).toArray())
console.log(ad.derivative(a))
// console.log(ad.derivative(c))
// console.log(roundArr(ad.derivative(x).toArray(),2));

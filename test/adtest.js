var assert = require('assert');
var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');
var _ = require('lodash');

var mat33 = [[1, 3, 5],
             [2, 4, 6],
             [9, 7, 3]];
var mat33T = [[1, 2, 9],
             [3, 4, 7],
             [5, 6, 3]];
var matdot = [[18, 18, 18],
              [24, 24, 24],
              [38, 38, 38]];
var inv = [[ -7.5, 6.5, -0.5],
             [ 12, -10.5, 1],
             [ -5.5, 5, -0.5]];

function roundArr (arr, sf) {
  var i = 0;
  while(i < arr.length){ 
    if (arr[i] instanceof Array) {
        console.log(arr[i])
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
    var x = tensortype === "tensor" ? new Tensor([5]).fill(1.5) : new THTensor([5]).fill(1.5);
    var y = x.clone().fill(0.2);
    var z = y.clone().fill(0.3);
    var d_1 = tensortype === "tensor" ? new Tensor([3,3]).fromArray(mat33) : new THTensor([3,3]).fromArray(mat33);
    var d2 = d_1.clone().fill(2);
    describe(type, function () {
        describe("Math Ops", function () {
            describe("Unary", function () {
                //Unary operators
                it('tanh', function () { assert.equal(Number(ad.scalar.tanh(1.5)).toFixed(3), 0.905) });  // 0.9051...
                it('tanh', function () { assert.deepEqual(roundArr(ad.tensor.tanh(x).toArray(), 3), fillArr(0.905, 5)) });
                it('neg', function () { assert.deepEqual(roundArr(ad.tensor.neg(x).toArray(), 1), fillArr(-1.5, 5)) });
                it('floor', function () { assert.deepEqual(roundArr(ad.tensor.floor(x).toArray(), 1), fillArr(1, 5)) });
                it('ceil', function () { assert.deepEqual(roundArr(ad.tensor.ceil(x).toArray(), 1), fillArr(2, 5)) });
                it('round', function () { assert.deepEqual(roundArr(ad.tensor.round(x).toArray(), 1), fillArr(2, 5)) });
                it('sqrt', function () { assert.deepEqual(roundArr(ad.tensor.sqrt(x).toArray(), 2), fillArr(1.22, 5)) });
                it('exp', function () { assert.deepEqual(roundArr(ad.tensor.exp(x).toArray(), 2), fillArr(4.48, 5)) });
                it('log', function () { assert.deepEqual(roundArr(ad.tensor.log(x).toArray(), 3), fillArr(0.405, 5)) });
                it('abs', function () { assert.deepEqual(roundArr(ad.tensor.abs(x).toArray(), 2), fillArr(1.5, 5)) });
                it('sin', function () { assert.deepEqual(roundArr(ad.tensor.sin(x).toArray(), 3), fillArr(0.997, 5)) });
                it('cos', function () { assert.deepEqual(roundArr(ad.tensor.cos(x).toArray(), 3), fillArr(0.071, 5)) });
                it('tan', function () { assert.deepEqual(roundArr(ad.tensor.tan(x).toArray(), 3), fillArr(14.101, 5)) });
                it('sigmoid', function () { assert.deepEqual(roundArr(ad.tensor.sigmoid(x).toArray(), 3), fillArr(0.818, 5)) });
            });
            describe ("Binary", function () {
                //Binary operators
                it('add', function () { assert.deepEqual(roundArr(ad.tensor.add(x, y).toArray(), 1), fillArr(1.7, 5)) });
                it('sub', function () { assert.deepEqual(roundArr(ad.tensor.sub(x, y).toArray(), 1), fillArr(1.3, 5)) });
                it('mul', function () { assert.deepEqual(roundArr(ad.tensor.mul(x, y).toArray(), 1), fillArr(0.3, 5)) });
                it('div', function () { assert.deepEqual(roundArr(ad.tensor.div(x, y).toArray(), 1), fillArr(7.5, 5)) });
                it('pow', function () { assert.deepEqual(roundArr(ad.tensor.pow(x, y).toArray(), 3), fillArr(1.084, 5)) });
                it('max', function () { assert.deepEqual(roundArr(ad.tensor.max(x, y).toArray(), 1), fillArr(1.5, 5)) });
                it('min', function () { assert.deepEqual(roundArr(ad.tensor.min(x, y).toArray(), 1), fillArr(0.2, 5)) });
                it('atan2', function () { assert.deepEqual(roundArr(ad.tensor.atan2(x, y).toArray(), 3), fillArr(1.438, 5)) });
            });
        });
        describe("Tensor Ops", function () {
            it('sumreduce', function () { assert.equal(ad.tensor.sumreduce(x), 7.5) });
            it('allreduce', function () { assert.ok(ad.tensor.allreduce(x)) });
            it('anyreduce', function () { assert.ok(ad.tensor.anyreduce(x)) });
            it('get', function () { assert.equal(ad.tensor.get(x, 1), 1.5) });
            it('toScalars', function () { assert.deepEqual(ad.tensor.toScalars(x), [1.5, 1.5, 1.5, 1.5, 1.5]) });
            it('fromScalars', function () { assert.deepEqual(ad.tensor.fromScalars([0, 1, 2, 3]).toArray(), [0, 1, 2, 3]) });
            it('range', function () { assert.deepEqual(ad.tensor.range(x, 0, 2).toArray(), [1.5, 1.5]) });
        });
        describe("Linear Alg Ops", function () {
            it('split', function () { assert.deepEqual(ad.tensor.split(x, [2,3])[0].toArray(), [1.5, 1.5]) });
            // rounding errors for concat but it works
            it('concat', function () { assert.deepEqual(roundArr(ad.tensor.concat([x,y]).toArray(), 1),
                            [1.5, 1.5, 1.5, 1.5, 1.5,
                            0.2, 0.2, 0.2, 0.2, 0.2]) });
//                             0.3, 0.3, 0.3, 0.3, 0.3]) });
            it('transpose', function () { assert.deepEqual(ad.tensor.transpose(d_1).toArray(), mat33T) });
            //   console.log(ad.tensor.diagonal(d2))
            it('inverse', function () { assert.deepEqual(roundArr(ad.tensor.inverse(d_1).toArray(), 2), inv) });
            it('determinant', function () { assert.equal(Number(ad.tensor.determinant(d_1)).toFixed(1), 4) });
            it('dot', function () { assert.deepEqual(ad.tensor.dot(d_1, d2).toArray(), matdot) });
            it('softmax', function () { assert.equal(Number(ad.tensor.softmax(x).toArray()[0]).toFixed(1), 0.2) });
        });
        });
}

run ("Torch");
run ("JS");
return


//====================
//TESTING reshjaped dot
//   var x= new THTensor([1, 4]).fill(1.5);
//   console.log(ad.tensor.range(x, 0, 2).toArray(), [1.5, 1.5])
// //   assert.deepEqual(ad.tensor.range(x, 0, 2).toArray(), [1.5, 1.5]))
// //   console.log(ad.tensor.dot(x_1,y_1).toArray())
//   return

//=======
//---------------
var t_in = ad.lift(d_1);
console.log(ad.tensor.relu(t_in).x.toArray());
//   return

//===========

// To compute derivatives, we must first turn input Numbers/Tensors into AD graph nodes
//    by 'lifting' them
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

// We can then retrieve the values and derivatives of different nodes
ad.value(scalarOut);  // 0.9051...
ad.derivative(scalarIn);  // 0.1807...

ad.isLifted(scalarIn);
ad.isLifted(1.5);

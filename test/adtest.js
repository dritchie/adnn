var assert = require('assert');
var ad = require('../ad');
var THTensor = require('../THTensor.js');
var Tensor = require('../tensor.js');
var _ = require('lodash');
var T = ad.tensor;

var vec = [1, 4, 3];

var vec4 = [[1], [4], [3], [2]];

var mat33 = [[1, 3, 5],
             [2, 4, 6],
             [9, 7, 3]];
var mat33T = [[1, 2, 9],
             [3, 4, 7],
             [5, 6, 3]];
var mat33D = [[1, 0, 0],
             [0, 4, 0],
             [0, 0, 3]];
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
    var v = tensortype === "tensor" ? new Tensor([3]).fromArray(vec) : new THTensor([3]).fromArray(vec);
    var s = tensortype === "tensor" ? new Tensor([4, 1]).fromArray(vec4) : new THTensor([4, 1]).fromArray(vec4);
    var d2 = d_1.clone().fill(2);
    describe(type, function () {
        describe("Math Ops", function () {
            describe("Unary", function () {
                //Unary operators
                it('tanh', function () { assert.equal(Number(ad.scalar.tanh(1.5)).toFixed(3), 0.905) });  // 0.9051...
                it('tanh', function () { assert.deepEqual(roundArr(T.tanh(x).toArray(), 3), fillArr(0.905, 5)) });
                it('neg', function () { assert.deepEqual(roundArr(T.neg(x).toArray(), 1), fillArr(-1.5, 5)) });
                it('floor', function () { assert.deepEqual(roundArr(T.floor(x).toArray(), 1), fillArr(1, 5)) });
                it('ceil', function () { assert.deepEqual(roundArr(T.ceil(x).toArray(), 1), fillArr(2, 5)) });
                it('round', function () { assert.deepEqual(roundArr(T.round(x).toArray(), 1), fillArr(2, 5)) });
                it('sqrt', function () { assert.deepEqual(roundArr(T.sqrt(x).toArray(), 2), fillArr(1.22, 5)) });
                it('exp', function () { assert.deepEqual(roundArr(T.exp(x).toArray(), 2), fillArr(4.48, 5)) });
                it('log', function () { assert.deepEqual(roundArr(T.log(x).toArray(), 3), fillArr(0.405, 5)) });
                it('abs', function () { assert.deepEqual(roundArr(T.abs(x).toArray(), 2), fillArr(1.5, 5)) });
                it('sin', function () { assert.deepEqual(roundArr(T.sin(x).toArray(), 3), fillArr(0.997, 5)) });
                it('cos', function () { assert.deepEqual(roundArr(T.cos(x).toArray(), 3), fillArr(0.071, 5)) });
                it('tan', function () { assert.deepEqual(roundArr(T.tan(x).toArray(), 3), fillArr(14.101, 5)) });
                it('sigmoid', function () { assert.deepEqual(roundArr(T.sigmoid(x).toArray(), 3), fillArr(0.818, 5)) });
            });
            describe ("Binary", function () {
                //Binary operators
                it('add', function () { assert.deepEqual(roundArr(T.add(x, y).toArray(), 1), fillArr(1.7, 5)) });
                it('sub', function () { assert.deepEqual(roundArr(T.sub(x, y).toArray(), 1), fillArr(1.3, 5)) });
                it('mul', function () { assert.deepEqual(roundArr(T.mul(x, y).toArray(), 1), fillArr(0.3, 5)) });
                it('div', function () { assert.deepEqual(roundArr(T.div(x, y).toArray(), 1), fillArr(7.5, 5)) });
                it('pow', function () { assert.deepEqual(roundArr(T.pow(x, y).toArray(), 3), fillArr(1.084, 5)) });
                it('max', function () { assert.deepEqual(roundArr(T.max(x, y).toArray(), 1), fillArr(1.5, 5)) });
                it('min', function () { assert.deepEqual(roundArr(T.min(x, y).toArray(), 1), fillArr(0.2, 5)) });
                it('atan2', function () { assert.deepEqual(roundArr(T.atan2(x, y).toArray(), 3), fillArr(1.438, 5)) });
            });
        });
        describe("Tensor Ops", function () {
            it('sumreduce', function () { assert.equal(T.sumreduce(x), 7.5) });
            it('allreduce', function () { assert.ok(T.allreduce(x)) });
            it('anyreduce', function () { assert.ok(T.anyreduce(x)) });
            it('get', function () { assert.equal(T.get(x, 1), 1.5) });
            it('reshape', function () { assert.deepEqual(T.reshape(s, [2,2]).toArray(), [[1, 4], [3, 2]]) });
            it('toScalars', function () { assert.deepEqual(T.toScalars(x), [1.5, 1.5, 1.5, 1.5, 1.5]) });
            it('fromScalars', function () { assert.deepEqual(T.fromScalars([0, 1, 2, 3]).toArray(), [0, 1, 2, 3]) });
            it('range', function () { assert.deepEqual(T.range(x, 0, 2).toArray(), [1.5, 1.5]) });
        });
        describe("Linear Alg Ops", function () {
            it('split', function () { assert.deepEqual(T.split(x, [2,3])[0].toArray(), [1.5, 1.5]) });
            // rounding errors for concat but it works
            it('concat', function () { assert.deepEqual(roundArr(T.concat([x,y]).toArray(), 1),
                            [1.5, 1.5, 1.5, 1.5, 1.5,
                            0.2, 0.2, 0.2, 0.2, 0.2]) });
//                             0.3, 0.3, 0.3, 0.3, 0.3]) });
            it('transpose', function () { assert.deepEqual(T.transpose(d_1).toArray(), mat33T) });
            // this takes a rank 1 vector and returns an nxn matrix with the diagonals as the vectors elements
            it('diagonal', function () { assert.deepEqual(T.diagonal(v).toArray(), mat33D) });
//          diagonal entries of matrix, returned as an nxn matrix
            it('diag', function () { assert.deepEqual(T.diag(d_1).toArray(), mat33D) });
            it('inverse', function () { assert.deepEqual(roundArr(T.inverse(d_1).toArray(), 2), inv) });
            it('determinant', function () { assert.equal(Number(T.determinant(d_1)).toFixed(1), 4) });
            it('dot', function () { assert.deepEqual(T.dot(d_1, d2).toArray(), matdot) });
            it('softmax', function () { assert.equal(Number(T.softmax(x).toArray()[0]).toFixed(1), 0.2) });
        });
    });
}

// run ("Torch");
run ("JS");

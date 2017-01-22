'use strict';

var assert = require('assert');
var utils = require('./utils.js');

//var ffi = require('ffi')
//var ref =  require('ref')
//var THTensor = torch.THFloatTensor
//var THStorage = torch.THFloatStorage
var THType = "Float"

var ffith = require('/Users/jpchen/jstorch/torch.js/TH.js')
var TH = ffith.TH

var arr_to_ls = function(dims){
  var dimension = 0;
  var size = TH.THLongStorage_newWithSize(dims.length).deref();
  TH.THLongStorage_fill(size.ref(), 0)
  for(var i=0; i < dims.length; i++) {
    // torch.THLongStorage.resize(size.ref(), dimension + 1);
    TH.THLongStorage_set(size.ref(), i, dims[i])
    // size.data[dimension] = dims[i];
    // dimension++;

    // console.log("s1:", size.data[i])
  }

  // console.log("dim size:", TH.THLongStorage_size(size.ref()), " true: ", dims.length)
  // console.log(size)
  // console.log("dim info:", TH.THLongStorage_size(size.ref()), " true: ", dims.length)
  return size;
}

Tensor.ls_to_array = function(ls) {
  var dims = []
  for(var i=0; i < ls.size; i++){
    dims.push(TH.THLongStorage_get(ls.ref(), i));
  }
  return dims
}

var f_arr_prod = function(dims) {
    var prod = 1;
    for(var i=0; i < dims.length; i++)
        prod *= dims[i];
    return prod;
}

// Can swap out different backing stores
function TypedArrayBackingStore(ArrayType) {
    return {
        new: function(n) { return new ArrayType(n); },
        set: function(tgt, src, offset) {
            tgt.set(src, offset);
        }
    }
}

Tensor.prototype.override = function(t_data, dims) {
  this.dims = dims || Tensor.ls_to_array(TH.THFloatTensor_newSizeOf(t_data.ref()).deref())
  this.length = f_arr_prod(this.dims)
  this.data = t_data
  this.ref = this.data.ref()
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
  if(!Array.isArray(dims))
    throw new Error("Tensor must have an array provided for construction");

  var i, j;
  var counter, tensor;
  var si = 0;
  var dimension = 0;
  var is_finished = 0;

  var size = arr_to_ls(dims)

  var prod = f_arr_prod(dims)
  this.dims = dims;
  this.length = prod

  // var tensor = THTensor.newWithSize1d(prod).deref();
  var tensor = TH.THFloatTensor_newWithSize(size.ref(), ref.NULL).deref();
  // var tensor = THTensor.newWithSize(size.ref(), ref.NULL).deref();
  // console.log("ref size:1 ", THTensor.nElement(tensor.ref()))
  this.data = tensor;
  this.ref = this.data.ref()
  this.type = THType;
  this._tensor_object = TH.THFloatTensor;
}


//original tensor
// function Tensor(dims) {
//     if(!Array.isArray(dims))
//     throw new Error("Tensor must have an array provided for construction");
//     this.dims = dims;
//     var size = 1;
//     var n = dims.length;
//     while (n--) size *= dims[n];
//     this.length = size;
//     this.data = BackingStore.new(size);
// }

Object.defineProperties(Tensor.prototype, {
    rank: { get: function() { return this.dims.length; } },
});

Tensor.prototype.reshape = function(dims) {
  var size = f_arr_prod(dims)
  assert(size === this.length, 'Tensor reshape invalid size');
  this.dims = dims;
  this.data = this.view(dims)
  this.ref = this.data.ref()
  return this
}

Tensor.prototype.sum = function(ix){
  if(ix == undefined || ix == null)
    return TH.THFloatTensor_sumall(this.ref)
  else{
    throw new Error("Sum across dimensions is not yet supported")
  }
}

Tensor.prototype.sumreduce = Tensor.prototype.sum

Tensor.prototype.view = function(dims) {
  return Tensor.view_tensor(this.data, dims)
}

Tensor.prototype.fill = function(val) {
    TH.THFloatTensor_fill(this.ref, val);
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

// Make this Tensor refer to the same backing store as other
Tensor.prototype.refCopy = function(other) {
    this.dims = other.dims;
    this.length = other.length;
    this.data = other.data;
    this.ref = this.data.ref()
    this.type = other.type;
    return this;
}

// Create a new Tensor object that refers to the same backing store
//    as this Tensor object
Tensor.prototype.refClone = function() {
    var t = Object.create(Tensor.prototype);
    return t.refCopy(this);
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

Tensor.create_empty_of_size = function(ts, TensorType) {
  //Supporting floats for now.
  //TensorType = TensorType || THFloatTensor
  return TH.THFlotTensor_newWithSize(Tensor.get_size(ts, TensorType).ref(), ref.NULL).deref()
}

Tensor.getSize = function(ts, TensorType) {
    // Supporting floats for now
    //TensorType = TensorType || THFloatTensor
    return TH.THFloatTensor_newSizeOf(ts).deref()
}

Tensor.prototype.size = function(ix) {
    if(ix != undefined)
        return TH.THFloatTensor_size(this.data.ref(), ix)
    else
        return Tensor.ls_to_array(Tensor.getSize(this.data.ref()))
};


Tensor.prototype.min = function() {
  return TH.THFloatTensor_minall(this.data.ref())
}
Tensor.prototype.minreduce = Tensor.prototype.min

Tensor.prototype.max = function() {
  return TH.THFloatTensor.maxall(this.data.ref())
}
Tensor.prototype.maxreduce = Tensor.prototype.max
// function toArrayRec(tensor, coords) {
//     if (coords.length === tensor.rank) {
//         return tensor.get(coords);
//     } else {
//         var dim = coords.length;
//         var arr = [];
//         for (var i = 0; i < tensor.dims[dim]; i++) {
//             arr.push(toArrayRec(tensor, coords.concat([i])));
//         }
//         return arr;
//     }
// }
// Tensor.prototype.toArray = function() {
//     return toArrayRec(this, []);
// };
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
    return this.toArray().toString();
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
        '   var x = this.data[n];',
        '   this.data[n] = ' + fncode + ';',
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
        '   var a = this.data[n];',
        '   this.data[n] = ' + fncode + ';',
        '}',
        'return this;'
    ].join('\n'));
    var fneqT = new Function('t', [
        'var n = this.data.length;',
        'while (n--) {',
        '   var a = this.data[n];',
        '   var b = t.data[n];',
        '   this.data[n] = ' + fncode + ';',
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
        '   var x = this.data[n];',
        '   accum = ' + fncode + ';',
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



Tensor.prototype.softmax = function() {
    // Find max elem
    var max = -Infinity;
    var n = this.data.length;
    while (n--) {
        max = Math.max(max, this.data[n]);
    }
    var t = new Tensor(this.dims);
    // Exponentiate, guard against overflow
    n = this.data.length;
    var sum = 0;
    while (n--) {
        t.data[n] = Math.exp(this.data[n] - max);
        sum += t.data[n];
    }
    // Normalize
    n = this.data.length;
    while (n--) {
        t.data[n] /= sum;
    }
    return t;
};

Tensor.prototype.transpose = function(ix, ix2) {
  var ccTensor = this.clone()
  if(ix == undefined)
    THFloatTensor_transpose(ccTensor.data.ref(), ref.NULL, 0, 1)
  else
    TH.THFloatTensor_transpose(ccTensor.data.ref(), ref.NULL, ix, ix2)
  return ccTensor
}
Tensor.prototype.T = Tensor.prototype.transpose
// Do the conservative thing, and return a copy for now.
// Tensor.prototype.transpose = function() {
//   assert.ok(this.rank === 2);
//   var h = this.dims[0];
//   var w = this.dims[1];
//   var y = new Tensor([w, h]);
//   for (var i = 0; i < h; i++) {
//     for (var j = 0; j < w; j++) {
//       y.data[j * h + i] = this.data[i * w + j];
//     }
//   }
//   return y;
// };

Tensor.prototype.diagonal = function() {
  assert.ok(this.rank === 2);
  assert.ok(this.dims[1] === 1);
  
  var etensor = Tensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_diag(etensor.ref(),this.data.ref(),0)
  var ccTensor = this.refClone()
  ccTensor.override(etensor, this.dims)
  return ccTensor
};

// Matrix inverse.
Tensor.prototype.inverse = function() {

  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);

  var etensor = Tensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_getri(etensor.ref(), this.data.ref())

  var ccTensor = this.refClone()
  ccTensor.override(etensor, this.dims)
  return ccTensor
};

// Determinant.
// Ported from numeric.js.
Tensor.prototype.determinant = function() {
  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);
  var n = this.dims[0];
  var ret = 1;

  var i, j, k;
  var Aj, Ai, alpha, temp, k1, k2, k3;

  var A = [];
  for (i = 0; i < n; i++) {
    Ai = new Float64Array(n);
    A.push(Ai);
    for (j = 0; j < n; j++) {
      Ai[j] = this.data[i * n + j];
    }
  }

  for (j = 0; j < n - 1; j++) {
    k = j;
    for (i = j + 1; i < n; i++) {
      if (Math.abs(A[i][j]) > Math.abs(A[k][j])) {
        k = i;
      }
    }
    if (k !== j) {
      temp = A[k];
      A[k] = A[j];
      A[j] = temp;
      ret *= -1;
    }
    Aj = A[j];
    for (i = j + 1; i < n; i++) {
      Ai = A[i];
      alpha = Ai[j] / Aj[j];
      for (k = j + 1; k < n - 1; k += 2) {
        k1 = k + 1;
        Ai[k] -= Aj[k] * alpha;
        Ai[k1] -= Aj[k1] * alpha;
      }
      if (k !== n) {
        Ai[k] -= Aj[k] * alpha;
      }
    }
    if (Aj[j] === 0) {
      return 0;
    }
    ret *= Aj[j];
  }
  return ret * A[j][j];
};

Tensor.prototype.dot = function(t) {
  var a = this, b = t;

  if (a.rank !== 2 || b.rank !== 2) {
    throw new Error('Inputs to dot should have rank = 2.');
  }
  if (a.dims[1] !== b.dims[0]) {
    throw new Error('Dimension mismatch in dot. Inputs have dimension ' + a.dims + ' and ' + b.dims + '.');
  }

  var l = a.dims[1];
  var h = a.dims[0], w = b.dims[1];
  var y = new Tensor([h, w]);

  for (var r = 0; r < h; r++) {
    for (var c = 0; c < w; c++) {
      var z = 0;
      for (var i = 0; i < l; i++) {
        z += a.data[r * l + i] * b.data[w * i + c];
      }
      y.data[r * w + c] = z;
    }
  }
  return y;
};

Tensor.prototype.cholesky = function() {
  assert.ok((this.rank === 2) && (this.dims[0] === this.dims[1]),
            'cholesky is only defined for square matrices.');

  var cc = Tensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_potrf(this.data.ref(), cc.ref())
  var ccTensor = this.refClone()
  ccTensor.override(cc, this.dims.slice(0))
  return ccTensor
};


module.exports = Tensor;




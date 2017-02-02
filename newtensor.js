'use strict';

var assert = require('assert');
var utils = require('./utils.js');

//var ffi = require('ffi')
var ref =  require('ref')
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
    TH.THLongStorage_set(size.ref(), i, dims[i])
    // size.data[dimension] = dims[i];
    // dimension++;

    // console.log("s1:", size.data[i])
    // if (i%1000==0){
    //   global.gc()
    // }
  }
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
    for(var i=0; i < dims.length; i++) {
        prod *= dims[i];
        // if (i%1000==0){
        //   global.gc()
        // }
      }
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
   o }
};


// The actual backing store we're using
var BackingStore = TypedArrayBackingStore(Float64Array);

function Tensor(dims) {
  if(!Array.isArray(dims))
    throw new Error("Tensor must have an array provided for construction");

  var size = arr_to_ls(dims)
  var prod = f_arr_prod(dims)
  this.dims = dims;
  this.length = prod

  // var tensor = THTensor.newWithSize1d(prod).deref();
  var tensor = TH.THFloatTensor_newWithSize(size.ref(), ref.NULL).deref();
  this.data = tensor;
  this.ref = this.data.ref()
  this.type = THType;
  return this;
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

Tensor.prototype.view = function(dims) {
  var rsize = arr_to_ls(dims);
  var nt = TH.THFloatTensor_new().deref()
  // console.log("orig", orig)
  TH.THFloatTensor_set(nt.ref(), orig.ref()) //orig.storage, orig.storageOffset, rsize.ref())
  return nt;
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
    while (n--){
       this.data[n] = utils.gaussianSample(0, scale);
    }
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

Tensor.is_equal = function(a,b) {
  if(a.length != b.length)
    return false
  for(var i=0; i < a.length; i++)
    if(a[i] != b[i])
      return false

  return true
}

Tensor.prototype.assert_size_equal = function(other, assert_msg) {
  if(typeof(other) == "number")
    return true
  else{
    var are_equal = Tensor.is_equal(other.dims, this.dims)
    assert.ok(are_equal, assert_msg)
    return are_equal
  }
}

// Gets or sets values of tensor, determined by the val_or_tensor arg
Tensor.get_set = function(js_tensor, coords, val_or_tensor) {
  var ndims = js_tensor.rank;
  var dfinal = ndims
  var cdim = 0;

  var o_tensor = js_tensor.data
  var tensor = TH.THFloatTensor_newWithTensor(o_tensor.ref()).deref();
  for(var dim = 0; dim < dfinal; dim++) {
    var pix = coords[dim]
    if(!Array.isArray(pix)) {
      pix = Math.floor(pix);
      if (pix < 0)
        pix = tensor.size[cdim] + pix + 1;
      if(!((pix >= 0) && (pix < tensor.size[cdim])))
        throw new Error("Index out of bounds.");
      if(ndims == 1){
        // Setting element
        if (val_or_tensor != undefined){
          if (typeof(val_or_tensor) != "number")
            throw new Error("Value being set needs to be number.");
          TH.THFloatStorage_set(tensor.storage, tensor.storageOffset+pix*tensor.stride[0], val_or_tensor);
          return;
        }
        else{
          var rval = TH.THFloatStorage_get(tensor.storage, tensor.storageOffset+pix*tensor.stride[0]);
          return rval;
        }
      }
      else {
        TH.THFloatTensor_select(tensor.ref(), ref.NULL, cdim, pix);
        ndims = TH.THFloatTensor_nDimension(tensor.ref());
      }
    }
    else if(typeof(pix) != "number") {
      // SAfety check
      tensor = null;
      throw new Error("Tensor index must be an Int or an Array of ints.");
    }
    else {
      //Array
      var ixarray = pix;
      var start = 0;
      var end = tensor.size[cdim]-1;
      if(ixarray.length > 0) {
        start = ixarray[0];
        end = start;
      }

      if(start < 0)
        start = tensor.size[cdim] + start + 1;
      if(!((start >= 0) && (start < tensor.size[cdim])))
        throw new Error("Index out of bounds");
      if(ixarray.length > 1)
        end = ixarray[1];
      if(end < 0)
        end = tensor.size[cdim] + end + 1;
      if(!((end >= 0) && (end < tensor.size[cdim])))
        throw new Error("Index out of bounds");
      if(end < start)
        throw new Error("Starting index cannot be after End.");
      TH.THFloatTensor_narrow(tensor.ref(), ref.NULL, cdim++, start, end-start+1);
      ndims = TH.THFloatTensor_nDimension(tensor.ref());
    }
  }
  // copy from the tensor value
  if (val_or_tensor) {
    //THFloatTensor['copy' + val_or_tensor.type](tensor.ref(), val_or_tensor.data.ref())
    TH.THLongStorage_copyFloat(tensor.ref(), val_or_tensor.data.ref());
  }
  return tensor
}

// These are slow; don't use them inside any hot loops (i.e. they're good for
//    debgugging/translating data to/from other formats, and not much else)
Tensor.prototype.get = function(coords) {
  if(coords.length > this.dims.length)
    throw new Error("Dimensions exceeded rank")
  var tensor = Tensor.get_set(this, coords)

  if(tensor == undefined || typeof(tensor) == "number")
    return tensor
  else {
    var tt_ref = this.refClone()
    tt_ref.override(tensor)
    return tt_ref
  }
};

Tensor.prototype.set = function(coords, val) {
  // val is a scalar or a tensor
  var tensor = Tensor.get_set(this, coords, val)
  if(tensor == undefined)
    return tensor
  // create a reference to tensor
  var tt_ref = this.refClone()
  tt_ref.override(tensor)
  return tt_ref
};

// Tensor.prototype.get = function(coords) {
//     var idx = 0;
//     var n = this.dims.length;
//     for (var i = 0; i < n; i++) {
//         idx = idx * this.dims[i] + coords[i];
//     }
//     return this.data[idx];
// };
// Tensor.prototype.set = function(coords, val) {
//     var idx = 0;
//     var n = this.dims.length;
//     for (var i = 0; i < n; i++) {
//         idx = idx * this.dims[i] + coords[i];
//     }
//     this.data[idx] = val;
// };

Tensor.create_empty_of_size = function(ts, TensorType) {
  //Supporting floats for now.
  //TensorType = TensorType || THFloatTensor
  return TH.THFloatTensor_newWithSize(Tensor.getSize(ts, TensorType).ref(), ref.NULL).deref()
}

Tensor.getSize = function(ts, TensorType) {
    // Supporting floats for now
    //TensorType = TensorType || THFloatTensor
    return TH.THFloatTensor_newSizeOf(ts).deref()
}

Tensor.prototype.size = function(ix) {
    if(ix != undefined)
        return TH.THFloatTensor_size(this.data.ref(), ix);
    else
        return Tensor.ls_to_array(Tensor.getSize(this.data.ref()));
};

Tensor.byte_sizeof = function(sz, ttype) {
  var bempty = TH.THFloatTensor_newWithSize(sz.ref(), ref.NULL).deref();
  // console.log("empty in habitat: ", bempty)
  return {empty: bempty, byte_tensor: bbtensor};
}

Tensor.byte_nonzero = function(ts, ttype) {
  var sz = Tensor.getSize(ts.ref());
  var tempty = TH.THFloatTensor_newWithSize(sz.ref(), ref.NULL).deref();
  TH.THFloatTensor_zero(tempty.ref());
  var b_obj = Tensor.byte_sizeof(sz, ttype);
  var bempty = b_obj.empty;
  var bbtensor = b_obj.byte_tensor;

  // fill byte tensor with not equals
  TH.THFloatTensor_neTensor(bempty.ref(), tempty.ref(), ts.ref())
  return bbtensor.sumall(bempty.ref())
}

Tensor.prototype.sum = function(ix) {
  if(ix == undefined || ix == null)
    return TH.THFloatTensor_sumall(this.ref);
  else{
    throw new Error("Sum across dimension not yet supported");
  }
}
Tensor.prototype.sumreduce = Tensor.prototype.sum

Tensor.prototype.min = function() {
  return TH.THFloatTensor_minall(this.data.ref());
}
Tensor.prototype.minreduce = Tensor.prototype.min;

Tensor.prototype.max = function() {
  return TH.THFloatTensor_maxall(this.data.ref());
}
Tensor.prototype.maxreduce = Tensor.prototype.max;

Tensor.prototype.all = function() {
  return Tensor.byte_nonzero(this.data, this.type) == this.length;
}
Tensor.prototype.allreduce = Tensor.prototype.all;

Tensor.prototype.any = function() {
  return Tensor.byte_nonzero(this.data, this.type) > 0;
}
Tensor.prototype.anyreduce = Tensor.prototype.any;


Tensor.atan2 = function(adata, bdata, not_in_place, mval)
{
  mval = mval || 1;
  var end_ref = adata.data;

  if(not_in_place)
    end_ref = Tensor.create_empty_of_size(adata.data.ref());

  if(typeof(bdata) == "number"){
    TH.THFloatTensor_add(end_ref.ref(), adata.data.ref(), bdata);
    bdata = {data: Tensor.create_empty_of_size(adata.data.ref())};
    TH.THFloatTensor_fill(bdata.data.ref(), bdata)
  }

  TH.THFloatTensor_atan2(end_ref.ref(), adata.data.ref(), bdata.data.ref())

  return end_ref
}

function toArrayRec(tensor, coords) {
    if (coords.length === tensor.rank) {
        return tensor.get(coords);
    } else {
        var dim = coords.length;
        var arr = [];
        for (var i = 0; i < tensor.dims[dim]; i++) {
            arr.push(toArrayRec(tensor, coords.concat([i])));
        }
        return arr;
    }
}

Tensor.prototype.toFlatArray = function () {
    var arr = [];
    if (this.rank === 1) {
      for (var i=0; i < this.dims[0]; ++i) {
        arr.push(TH.THFloatTensor_get1d(this.data.ref(), i));
      }
      return arr;
    }
    else if (this.rank === 2) {
      for (var i=0; i < this.dims[0]; ++i) {
        for (var j=0; j < this.dims[1]; ++j) {
          arr.push(TH.THFloatTensor_get2d(this.data.ref(), i, j));
        }
      }
      return arr;
    }
    throw new Error('Tensors must have rank = 1 or 2');
}

Tensor.prototype.toArray = function() {
    return toArrayRec(this, []);
};

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

// Tensor.prototype.toFlatArray = function() {
//     return Array.prototype.slice.call(this.data);
// }
Tensor.prototype.fromFlatArray = function(arr) {
    BackingStore.set(this.data, arr, 0);
    return this;
}

function addUnaryMethod(name) {
    //need to differentiate between in-place and non in-place operations
    Tensor[name] = new Function('TH', 'Tensor', [
    'return function(adata, inplace){',
    'var end_ref = adata.data',
    'if(!inplace) {',
      'end_ref = Tensor.create_empty_of_size(adata.data.ref())',
    '}',
    // operation in place please
    'TH.THFloatTensor_' + name + '(end_ref.ref(), adata.data.ref())',
    'return end_ref}'
  ].join('\n'))(TH, Tensor);
    // var fneq = new Function([
    //     'var n = this.data.length;',
    //     'while (n--) {',
    //     '   var x = this.data[n];',
    //     '   this.data[n] = ' + fncode + ';',
    //     '}',
    //     'return this;'
    // ].join('\n'));
    // Tensor.prototype[name + 'eq'] = fneq;
    // Tensor.prototype[name] = function() {
    //     var nt = this.clone();
    //     return fneq.call(nt);
    // };
}

function addUnaryPrototype(name){
  // Use method generated in addUnaryMethod()
  var fn_inplace = new Function('Tensor', [
      'return function(){',
      'Tensor.' + name + '(this, true)',
      'return this}'
  ].join('\n'))(Tensor);
  //clone if not in-place
  var fn_notinplace = new Function('Tensor', [
      'return function(){',
      'var atensor = Tensor.' + name + '(this, false)',
      'var cc = this.refClone()',
      'cc.override(atensor, this.dims.slice(0))',
      'return cc}'
  ].join('\n'))(Tensor);

  Tensor.prototype[name + 'eq'] = fn_inplace;
  Tensor.prototype[name] = fn_notinplace
}

function addOperationOrComponentOpMethod(name, comp_method, no_mval) {
  Tensor[name] = new Function('TH', 'Tensor', [
    'return function(adata, bdata, not_in_place, mval){',
    'mval = mval || 1',
    'var end_ref = adata.data',

    // if not in place, we have to add
    'if(not_in_place)',
    '{',
      'end_ref = Tensor.create_empty_of_size(adata.data.ref())',
    '}',

    'if(typeof(bdata) == "number")',
      'TH.THFloatTensor_' + name + '(end_ref.ref(), adata.data.ref(), mval*bdata)',
    'else',
      'TH.THFloatTensor_' + comp_method + '(end_ref.ref(), adata.data.ref(), ' + (no_mval ? '' : 'mval, ') + 'bdata.data.ref())',
    'return end_ref}'
  ].join('\n'))(TH, Tensor);
}

function addBinaryMethod(name, mulval) {
  mulval = mulval || 1

  var fn_inplace = new Function('Tensor', [
      'return function(c_or_tensor){',
      'this.assert_size_equal(c_or_tensor, "C' + name + ' must be equal sizes")',
      'Tensor.' + name + '(this, c_or_tensor, false, ' + mulval + ')',
      'return this}'
  ].join('\n'))(Tensor);

 var fn_notinplace = new Function('Tensor', [
      'return function(c_or_tensor){',
      'this.assert_size_equal(c_or_tensor, "C' + name + ' must be equal sizes")',
      'var atensor = Tensor.' + name + '(this, c_or_tensor, true, ' + mulval + ')',
      'var cc = this.refClone()',
      ' cc.override(atensor, this.dims.slice(0))',
      'return cc}'
  ].join('\n'))(Tensor);

  Tensor.prototype[name + 'eq'] = fn_inplace;
  Tensor.prototype[name] = fn_notinplace
}
// function addBinaryMethod(name, fncode) {
//     var fneqS = new Function('s', [
//         'var n = this.data.length;',
//         'var b = s;',
//         'while (n--) {',
//         '   var a = this.data[n];',
//         '   this.data[n] = ' + name + ';',
//         '}',
//         'return this;'
//     ].join('\n'));
//     var fneqT = new Function('t', [
//         'var n = this.data.length;',
//         'while (n--) {',
//         '   var a = this.data[n];',
//         '   var b = t.data[n];',
//         '   this.data[n] = ' + fncode + ';',
//         '}',
//         'return this;'
//     ].join('\n'));

//     var fneq = function(x) {
//         if (x.constructor === Tensor)
//             return fneqT.call(this, x);
//         else
//             return fneqS.call(this, x);
//     }
//     Tensor.prototype[name + 'eq'] = fneq;
//     Tensor.prototype[name] = function(x) {
//         var nt = this.clone();
//         return fneq.call(nt, x);
//     };
// }

// function addReduction(name, initcode, fncode) {
//     Tensor.prototype[name+'reduce'] = new Function([
//         'var accum = ' + initcode + ';',
//         'var n = this.data.length;',
//         'while (n--) {',
//         '   var x = this.data[n];',
//         '   accum = ' + fncode + ';',
//         '}',
//         'return accum;'
//     ].join('\n'));
// }

//sub not needed since it is -1 of add
var arith = ['add', 'mul', 'div', 'pow'];

function createPrototype(name, isUnary, comp_meth, no_mval) {
    if (isUnary) {
        addUnaryMethod(name);
        addUnaryPrototype(name);
    } else {
        if (arith.indexOf(name) > -1){
          addOperationOrComponentOpMethod(name, comp_meth, no_mval);
        }
        addBinaryMethod(name);
    }
}

createPrototype('neg', true);
createPrototype('round', true);
createPrototype('log', true);
createPrototype('exp', true);
createPrototype('sqrt', true);
createPrototype('abs', true);
createPrototype('ceil', true);
createPrototype('floor', true);
createPrototype('cos', true);
createPrototype('sin', true);
createPrototype('tan', true);
createPrototype('acos', true);
createPrototype('asin', true);
createPrototype('atan', true);
createPrototype('cosh', true);
createPrototype('sinh', true);
createPrototype('tanh', true);
createPrototype('sigmoid', true);

// Warning: These do not exist in THm impl
createPrototype('acosh', true);
createPrototype('asinh',  true);
createPrototype('atanh', true);

//TODO: impl
createPrototype('isFinite', true);
createPrototype('isNaN', true);
createPrototype('invert', true);
createPrototype('pseudoinvert', true);

createPrototype('add', false, 'cadd');
addBinaryMethod('sub', false, -1);
createPrototype('mul', false, "cmul", true);
createPrototype('div', false, "cdiv", true);
createPrototype('fmod', false);
createPrototype('pow', false, "cpow", true);
createPrototype('atan2', false);
//TODO: torch method name
createPrototype('eq', false);
createPrototype('ne', false);
createPrototype('gt', false);
createPrototype('ge', false);
createPrototype('lt', false);
createPrototype('le', false);


//TODO: implement
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
    TH.THFloatTensor_transpose(ccTensor.data.ref(), ref.NULL, 0, 1)
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

// Determinant - cannot port from TH
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
  var t_for_mul = TH.THFloatTensor_new().deref();
  TH.THFloatTensor_resize2d(t_for_mul.ref(), a.dims[0], b.dims[1]);
  var beta = 0, alpha = 1;
  TH.THFloatTensor_addmm(t_for_mul.ref(), beta, t_for_mul.ref(), alpha, a.data.ref(), b.data.ref());
  var mm_tensor = a.refClone();
  mm_tensor.override(t_for_mul, [a.dims[0], b.dims[1]]);
  return mm_tensor;
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




'use strict';

var assert = require('assert');
var utils = require('./utils.js');

var ffi = require('ffi')
var ref =  require('ref')
var THType = "Float" // supports Floats by default

var ffith = require('/Users/jpchen/jstorch/torch.js/TH.js')
var TH = ffith.TH

var arr_to_ls = function(dims){
  var dimension = 0;
  var size = TH.THLongStorage_newWithSize(dims.length).deref();
  TH.THLongStorage_fill(size.ref(), 0)
  for(var i=0; i < dims.length; i++) {
    TH.THLongStorage_set(size.ref(), i, dims[i]);
  }
  return size;
}

THTensor.ls_to_array = function(ls) {
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
      }
    return prod;
}

THTensor.prototype.override = function(t_data, dims, ttype) {
  this.dims = dims || THTensor.ls_to_array(TH.THFloatTensor_newSizeOf(t_data.ref()).deref());
  this.length = f_arr_prod(this.dims);
  this.data = t_data;
  if (ttype === "Byte")
    this.ref = this.data.ref;
  else {
    this.ref = this.data.ref();
  }
}

// The actual backing store we're using
// var BackingStore = TypedArrayBackingStore(Float64Array);

function THTensor(dims) {
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
  this.ffi = TH
  this.type = THType;
  return this;
}

Object.defineProperties(THTensor.prototype, {
    rank: { get: function() { return this.dims.length; } },
});

THTensor.prototype.reshape = function(dims) {
  var size = f_arr_prod(dims)
  assert(size === this.length, 'Tensor reshape invalid size');
  this.dims = dims;
  var ls = arr_to_ls(dims)
  var ccTensor = new THTensor(dims);
  //ccTensor.override(ccTensor, dims);
  TH.THFloatTensor_reshape(ccTensor.ref, this.data.ref(), ls.ref())
  // this.data = this.view(dims);
  // this.ref = this.data.ref();
  return ccTensor;
}

THTensor.prototype.view = function(dims) {
  var rsize = arr_to_ls(dims);
  var nt = TH.THFloatTensor_new().deref()
  // console.log("orig", orig)
  TH.THFloatTensor_set(nt.ref(), this.data.ref()) //orig.storage, orig.storageOffset, rsize.ref())
  return nt;
}

THTensor.prototype.fill = function(val) {
    TH.THFloatTensor_fill(this.ref, val);
    return this;
};

THTensor.prototype.zero = function() {
    return this.fill(0);
};

// Adapted from:
//    https://github.com/karpathy/convnetjs/blob/master/src/convnet_vol.js
THTensor.prototype.fillRandom = function() {
    var scale = 1/this.length;
    return this.applyFn(function(val) {
      return utils.gaussianSample(0, scale);
 });
}

THTensor.prototype.gaussian = function(mu, sigma) {
    return this.applyFn(function(val) {
      return utils.gaussianSample(mu, sigma);
 });
}

THTensor.prototype.diagCovGaussian = function(mu, sigma) {
    var cc = THTensor.create_empty_of_size(this.data.ref());
    var ccTensor = this.clone();
    ccTensor.override(cc, this.dims);
    TH.THFloatTensor_randn(ccTensor.ref);
    ccTensor.addeq(mu).muleq(sigma)
    return ccTensor;
}

THTensor.prototype.logGamma = function() {
    return this.applyFn(function(val) {
      return utils.logGamma(val);
 });
}

THTensor.prototype.copy = function(other, offset) {
    offset = offset || 0;

    if(offset != 0)
      throw new Error("Offset copying not yet supported")

    // TODO: erroring wrong size
    //TH.THFloatTensor['copy' + ttype](this.data.ref(), other.data.ref())
    TH.THFloatTensor_copyFloat(this.data.ref(), other.data.ref());
    return this;
};

// Slow Copy of array 
THTensor.prototype.slowCopy = function(other) {
    this.fromArray(other.toArray());
    return this;
};

THTensor.prototype.clone = function() {
    var copy = new THTensor(this.dims);
    // return copy.slowCopy(this);
    return copy.copy(this);
};

// Make this Tensor refer to the same backing store as other
THTensor.prototype.refCopy = function(other) {
    this.dims = other.dims;
    this.length = other.length;
    this.data = other.data;
    this.ref = this.data.ref();
    this.type = other.type;
    return this;
}

// Create a new Tensor object that refers to the same backing store
//    as this Tensor object
THTensor.prototype.refClone = function() {
    var t = Object.create(THTensor.prototype);
    return t.refCopy(this);
};

THTensor.arr_is_equal = function(a,b) {
  if(a.length != b.length)
    return false
  for(var i=0; i < a.length; i++)
    if(a[i] != b[i])
      return false

  return true
}

THTensor.prototype.assert_size_equal = function(other, assert_msg) {
  if(typeof(other) == "number")
    return true;
  else {
    var are_equal = THTensor.arr_is_equal(other.dims, this.dims);
    assert.ok(are_equal, assert_msg);
    return are_equal;
  }
}

// Gets or sets values of tensor, determined by the val_or_tensor arg
THTensor.get_set = function(js_tensor, coords, val_or_tensor) {
  var ndims = js_tensor.rank;
  var dfinal = ndims
  var cdim = 0;

  var o_tensor = js_tensor.data.ref instanceof Function ? js_tensor.data.ref() : js_tensor.data.ref;
  var tensor = TH.THFloatTensor_newWithTensor(o_tensor).deref();
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
  return tensor;
}

THTensor.prototype.narrow = function(dim, index, size) {
  var cc = THTensor.create_empty_of_size(this.data.ref());
  var ccTensor = this.clone();
  var dims = this.dims.slice();
  dims[dim] = size;
  ccTensor.override(cc, dims);
  TH.THFloatTensor_narrow(ccTensor.ref, this.data.ref(), dim, index, size);
  return ccTensor;
}

THTensor.prototype.range = function(start, end) {
  //hardcoded dim as 1
  var dim = 1;
  return this.narrow(dim, start, end - start);
}

// These are slow; don't use them inside any hot loops (i.e. they're good for
//    debgugging/translating data to/from other formats, and not much else)
THTensor.prototype.get = function(coords) {
  if(coords.length > this.dims.length)
    throw new Error("Dimensions exceeded rank")
  var tensor = THTensor.get_set(this, coords)

  if(tensor == undefined || typeof(tensor) == "number")
    return tensor
  else {
    var tt_ref = this.refClone()
    tt_ref.override(tensor)
    return tt_ref
  }
};

THTensor.prototype.set = function(coords, val) {
  // val is a scalar or a tensor
  var tensor = THTensor.get_set(this, coords, val);
  if(tensor == undefined)
    return tensor;
  // create a reference to tensor
  var tt_ref = this.refClone();
  tt_ref.override(tensor);
  return tt_ref;
};

THTensor.create_empty_of_size = function(ts, TensorType) {
  //Supporting floats for now.
  //TensorType = TensorType || THFloatTensor
  return TH.THFloatTensor_newWithSize(THTensor.getSize(ts, TensorType).ref(), ref.NULL).deref()
}

THTensor.getSize = function(ts, TensorType) {
    // Supporting floats for now
    //TensorType = TensorType || THFloatTensor
    return TH.THFloatTensor_newSizeOf(ts).deref()
}

THTensor.prototype.size = function(ix) {
    if(ix != undefined)
        return TH.THFloatTensor_size(this.data.ref(), ix);
    else
        return THTensor.ls_to_array(THTensor.getSize(this.data.ref()));
};

THTensor.byte_sizeof = function(sz, ttype) {
  return TH.THByteTensor_newWithSize(sz.ref(), ref.NULL).deref();
  //return {empty: bempty};
}

// Checks if a byte-tensor has non-zero elements
THTensor.byte_nonzero = function(ts, ttype) {
  var sz = THTensor.getSize(ts.ref());
  var tempty = TH.THFloatTensor_newWithSize(sz.ref(), ref.NULL).deref();
  TH.THFloatTensor_zero(tempty.ref());
  var bempty = THTensor.byte_sizeof(sz, ttype);

  // fill byte tensor with not equals
  TH.THFloatTensor_neTensor(bempty.ref(), tempty.ref(), ts.ref())
  return TH.THByteTensor_sumall(bempty.ref())
}

THTensor.byte_comparison = function(byte_comp_fct) {
  return function(adata, bdata, not_in_place, mval){
    assert.ok(not_in_place, "Cannot compare in-place equality");
    var sz = THTensor.getSize(adata.data.ref());
    var method = "THFloatTensor_" + byte_comp_fct;
    var tcompare = bdata.data;
    if (typeof(bdata) == "number") {
      tcompare = TH.THFloatTensor_newWithSize(sz.ref(), ref.NULL).deref();
      TH.THFloatTensor_fill(tcompare.ref(), bdata);
    } else {
      assert.ok(adata.type === bdata.type, "Checking tensor equal must be of same tensor type");
    }

    var bempty = THTensor.byte_sizeof(sz, adata.type);
    TH[method](bempty.ref(), adata.data.ref(), tcompare.ref());
    return bempty;
  }
}

THTensor.prototype.sum = function(ix) {
  if(ix == undefined || ix == null)
    return TH.THFloatTensor_sumall(this.ref);
  else{
    throw new Error("Sum across dimension not yet supported");
  }
}
THTensor.prototype.sumreduce = THTensor.prototype.sum

THTensor.prototype.minreduce = function() {
  return TH.THFloatTensor_minall(this.data.ref());
}

THTensor.prototype.min = function(other) {
  var cc = THTensor.create_empty_of_size(this.data.ref());
  var ccTensor = this.clone();
  ccTensor.override(cc, this.dims);
  TH.THFloatTensor_cmin(ccTensor.ref, this.ref, other.ref);
  return ccTensor;
}
THTensor.prototype.cmin = THTensor.prototype.min;

THTensor.prototype.maxreduce = function() {
  return TH.THFloatTensor_maxall(this.data.ref());
}

THTensor.prototype.max = function(other) {
  var cc = THTensor.create_empty_of_size(this.data.ref());
  var ccTensor = this.clone();
  ccTensor.override(cc, this.dims);
  TH.THFloatTensor_cmax(ccTensor.ref, this.ref, other.ref);
  return ccTensor;
}
THTensor.prototype.cmax = THTensor.prototype.max;

THTensor.prototype.allreduce = function() {
  return THTensor.byte_nonzero(this.data, this.type) == this.length;
}
THTensor.prototype.all = THTensor.prototype.allreduce;

THTensor.prototype.anyreduce = function() {
  return THTensor.byte_nonzero(this.data, this.type) > 0;
}
THTensor.prototype.any = THTensor.prototype.anyreduce;

THTensor.prototype.mod = function() {
  throw new Error("Mod not supported in torch, ergo no support yet.");
}

THTensor.prototype.modeq = function() {
  throw new Error("Mod not supported in torch, ergo no support yet.");
}

THTensor.atan2 = function(adata, bdata, not_in_place, mval) {
  mval = mval || 1;
  var end_ref = adata.data;

  if(not_in_place)
    end_ref = THTensor.create_empty_of_size(adata.data.ref());

  if(typeof(bdata) == "number"){
    TH.THFloatTensor_add(end_ref.ref(), adata.data.ref(), bdata);
    bdata = {data: THTensor.create_empty_of_size(adata.data.ref())};
    TH.THFloatTensor_fill(bdata.data.ref(), bdata);
  }

  TH.THFloatTensor_atan2(end_ref.ref(), adata.data.ref(), bdata.data.ref());

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

THTensor.prototype.printByteArray = function () {
    var arr = [];
    if (this.rank === 1) {
      for (var i=0; i < this.dims[0]; ++i) {
        arr.push(TH.THByteTensor_get1d(this.data.ref(), i));
      }
      return arr;
    }
    else if (this.rank === 2) {
      for (var i=0; i < this.dims[0]; ++i) {
        for (var j=0; j < this.dims[1]; ++j) {
          arr.push(TH.THByteTensor_get2d(this.data.ref(), i, j));
        }
      }
      return arr;
    }
    throw new Error('Tensors must have rank = 1 or 2');
}

THTensor.prototype.toFlatArray = function () {
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

THTensor.prototype.toArray = function() {
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
THTensor.prototype.fromArray = function(arr) {
    if (arr.length != this.dims[0])
      throw new Error('Array length must match with tensor length');
    if (arr.length > 256)
      return this.fromLargeArray(arr)
    fromArrayRec(this, [], arr);
    return this;
};

//concatenating these arrays for now
// this is VERY slow
THTensor.prototype.fromLargeArray = function (arr) {
  var n = arr.length;
  var rem = n % 255;
  var iter = Math.floor(n / 255)
  var out = []
  var arrindex = 0
  for (var j = 0; j < iter; j++) {
    var t = new THTensor([255])
    for (var i = 0; i < 255; i++) {
      t.set([i], arr[arrindex++]);
    }
    out.push(t);
  }
  var t = new THTensor([rem]);
  for (var i = 0; i < rem; i++){
    t.set([i], arr[arrindex++]);
  }
  out.push(t);
  var outten = out[0];
  for (var x = 1; x < out.length; x++){
    outten = outten.concat([out[x]]);
  }
  return outten;
}

THTensor.prototype.toString = function() {
    return this.toArray().toString();
};

THTensor.prototype.fromFlatArray = function(arr) {
    BackingStore.set(this.data, arr, 0);
    return this;
}

THTensor.prototype.applyFn = function (cb) {
  //eventually take any tensor type passed in
  var callback = ffi.Callback('float', ['float'], cb);
  TH.THFloatTensor_fctapply(this.data.ref(), callback);
  return this;
}

THTensor.prototype.concat = function (args, i) {
  // concatenating along 1 dimension for 1d Tensors
  if (i === null)
    i = 0;
  var n = args.length;
  assert.ok(n === 1);
  var size = this.length + args[0].length;
  var ccTensor = new THTensor([size])
  TH.THFloatTensor_cat(ccTensor.data.ref(), this.data.ref(), args[0].data.ref(), i);
  return ccTensor;
}

function addUnaryMethod(name) {
    //need to differentiate between in-place and non in-place operations
    THTensor[name] = new Function('TH', 'THTensor', [
    'return function(adata, notinplace) {',
    'var end_ref = adata.data',
    'if (notinplace) {',
      'end_ref = THTensor.create_empty_of_size(adata.data.ref())',
    '}',
    // operation in place please
    'TH.THFloatTensor_' + name + '(end_ref.ref(), adata.data.ref())',
    'return end_ref; }'
  ].join('\n'))(TH, THTensor);
}

function addUnaryPrototype(name){
  // Use method generated in addUnaryMethod()
  var fn_inplace = new Function('THTensor', [
      'return function(){',
      'THTensor.' + name + '(this, false)',
      'return this; }'
  ].join('\n'))(THTensor);
  //clone if not in-place
  var fn_notinplace = new Function('THTensor', [
      'return function() {',
      'var atensor = THTensor.' + name + '(this, true)',
      'var cc = this.refClone()',
      'cc.override(atensor, this.dims.slice(0))',
      'return cc }'
  ].join('\n'))(THTensor);

  THTensor.prototype[name + 'eq'] = fn_inplace;
  THTensor.prototype[name] = fn_notinplace
}

function addOperationOrComponentOpMethod(name, comp_method, no_mval) {
  THTensor[name] = new Function('TH', 'THTensor', [
    'return function(adata, bdata, not_in_place, mval) {',
    'mval = mval || 1;',
    'var end_ref = adata.data;',

    // if not in place, we have to add
    'if (not_in_place) {',
      'end_ref = THTensor.create_empty_of_size(adata.data.ref());',
    '}',

    'if (typeof(bdata) == "number")',
      'TH.THFloatTensor_' + name + '(end_ref.ref(), adata.data.ref(), mval * bdata)',
    'else {',
      // 'console.log(adata)',
      // 'console.log(bdata)',
      //'bref = typeof bdata.data.ref === "function" ? bdata.data.ref() : bdata.data.ref;',
      'TH.THFloatTensor_' + comp_method + '(end_ref.ref(), adata.data.ref(), ' + (no_mval ? '' : 'mval, ') + 'bdata.data.ref());',
      '}',
    'return end_ref; }'
  ].join('\n'))(TH, THTensor);
}

function addBinaryMethod(name, mulval, isbyte) {
  mulval = mulval || 1

  var fn_inplace = new Function('THTensor', [
      'return function(c_or_tensor){',
      'this.assert_size_equal(c_or_tensor, "C' + name + ' must be equal sizes")',
      'THTensor.' + name + '(this, c_or_tensor, false, ' + mulval + ')',
      'return this; }'
  ].join('\n'))(THTensor);

 var fn_notinplace = new Function('THTensor', [
      'return function(c_or_tensor){',
      'this.assert_size_equal(c_or_tensor, "C' + name + ' must be equal sizes")',
      'var atensor = THTensor.' + name + '(this, c_or_tensor, true, ' + mulval + ')',
      'var cc = this.refClone()',
      ' cc.override(atensor, this.dims.slice(0), "'+ isbyte +'")',
      'return cc; }'
  ].join('\n'))(THTensor);

  THTensor.prototype[name + 'eq'] = fn_inplace;
  THTensor.prototype[name] = fn_notinplace
}

// Adding sub because method name not overloaded in addbinarymethod
var arith = ['add', 'sub', 'mul', 'div', 'pow'];

function createPrototype(name, isUnary, comp_meth, no_mval, isbyte) {
    if (isUnary) {
        addUnaryMethod(name);
        addUnaryPrototype(name);
    } else {
        if (arith.indexOf(name) > -1) {
          addOperationOrComponentOpMethod(name, comp_meth, no_mval);
        }
        addBinaryMethod(name, no_mval, isbyte);
    }
}

// Unary prototypes
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

// Warning: These do not exist in THm impl
createPrototype('acosh', true);
createPrototype('asinh',  true);
createPrototype('atanh', true);

// Binary Prototypes
createPrototype('add', false, 'cadd');
//addBinaryMethod('sub', false, -1);
createPrototype('sub', false, 'csub');
createPrototype('mul', false, "cmul", true);
createPrototype('div', false, "cdiv", true);
createPrototype('fmod', false);
createPrototype('pow', false, "cpow", true);
createPrototype('atan2', false);

// These output a ByteTensor with the same dims as the input
THTensor.eq = THTensor.byte_comparison("eqTensor")
THTensor.ne = THTensor.byte_comparison("neTensor")
THTensor.gt = THTensor.byte_comparison("gtTensor")
THTensor.ge = THTensor.byte_comparison("geTensor")
THTensor.lt = THTensor.byte_comparison("ltTensor")
THTensor.le = THTensor.byte_comparison("leTensor")

createPrototype('eq', false, null, null, "Byte");
createPrototype('ne', false, null, null, "Byte");
createPrototype('gt', false, null, null, "Byte");
createPrototype('ge', false, null, null, "Byte");
createPrototype('lt', false, null, null, "Byte");
createPrototype('le', false, null, null, "Byte");

/*
 * The below functions are not supported by Torch
 */

// acosh
THTensor.prototype.acosheq = function() {
  //return Math.log(x + Math.sqrt(x * x - 1));
  var xx = this.clone().muleq(this).addeq(-1);
  return this.addeq(xx).logeq();
}

THTensor.prototype.acosh = function() {
  var cc = this.clone();
  cc.acosheq();
  return cc;
}

// asinh
THTensor.prototype.asinheq = function() {
  //return Math.log(x + Math.sqrt(x * x + 1));
  var xx = this.clone().muleq(this).addeq(1);
  return this.addeq(xx).logeq();
}

THTensor.prototype.asinh = function() {
  var cc = this.clone();
  cc.asinheq();
  return cc;
}

// atanh
THTensor.prototype.atanheq = function() {
  //Math.log((1+x)/(1-x)) / 2;
  var negxone = this.neg().addeq(1)
  return this.addeq(1).diveq(negxone).logeq().diveq(2)
}

THTensor.prototype.atanh = function() {
  var cc = this.clone();
  cc.atanheq();
  return cc;
}

THTensor.prototype.inverteq = function() {
  // '1 / x' 
  var cc = THTensor.create_empty_of_size(this.data.ref());
  var ccTensor = this.refClone();
  ccTensor.override(cc, this.dims);
  ccTensor.fill(1);
  ccTensor.diveq(this);
  this.copy(ccTensor);
  return this;
}

THTensor.prototype.invert = function() {
  var cc = this.clone();
  cc.inverteq();
  return cc;
}

THTensor.prototype.sigmoideq = function() {
  // 1 / (1 + Math.exp(-x)))
  return this.negeq().expeq().addeq(1).inverteq()
}

THTensor.prototype.sigmoid = function() {
  var cc = this.clone();
  cc.sigmoideq();
  return cc;
}

THTensor.prototype.isFiniteeq = function() {
  return this.applyFn(function(val) {
    return isFinite(val) ? 1.0 : 0.0;
  })
}

THTensor.prototype.isFinite = function() {
  var cc = this.clone();
  cc.isFiniteeq();
  return cc;
}

THTensor.prototype.isNaNeq = function() {
  return this.applyFn(function(val) {
    return isNaN(val) ? 1.0 : 0.0;
  });
}

THTensor.prototype.isNaN = function() {
  var cc = this.clone();
  cc.isNaNeq();
  return cc;
}

THTensor.prototype.pseudoinverteq = function() {
  return this.applyFn(function(val) {
    return val == 0 ? 0 : 1/val;
  });
}

THTensor.prototype.pseudoinvert = function() {
  var cc = this.clone();
  cc.pseudoinverteq();
  return cc;
}

// In-place softmax
THTensor.prototype.softmaxeq = function() {
  var max = this.maxreduce()
  // Don't clone
  var cc = this.addeq(-max).expeq()
  var sum = cc.sum()
  cc.diveq(sum)
  return this
};

THTensor.prototype.softmax = function() {
  // Find max elem
  var max = this.maxreduce()
  // clone it, subtract the max, then exponentiate
  var cc = this.clone().addeq(-max).expeq()
  var sum = cc.sum()
  // normalize
  cc.diveq(sum)
  return cc
};

THTensor.prototype.transpose = function(ix, ix2) {
  var ccTensor = this.clone()
  if(ix == undefined)
    TH.THFloatTensor_transpose(ccTensor.data.ref(), ref.NULL, 0, 1)
  else
    TH.THFloatTensor_transpose(ccTensor.data.ref(), ref.NULL, ix, ix2)
  return ccTensor
}
THTensor.prototype.T = THTensor.prototype.transpose

THTensor.prototype.diagonal = function() {
  assert.ok(this.rank === 2);
  //assert.ok(this.dims[1] === 1);
  
  var etensor = THTensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_diag(etensor.ref(),this.data.ref(),0)
  var ccTensor = this.refClone()
  ccTensor.override(etensor, this.dims)
  return ccTensor
};

//NN functions
THTensor.prototype.relueq = function() {
  return this.applyFn(function(val) {
    return  val < 0 ? 0 : val;
  });
}

THTensor.prototype.relu = function() {
  var cc = this.clone();
  cc.relueq();
  return cc;
}

THTensor.prototype.lrelueq = function() {
  //leakyness = 100
  return this.applyFn(function(val) {
     return  val < val/100 ? 0 : val;
  });
}

THTensor.prototype.lrelu = function() {
  var cc = this.clone();
  cc.lrelueq();
  return cc;
}

// Matrix inverse.
THTensor.prototype.inverse = function() {

  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);

  var etensor = THTensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_getri(etensor.ref(), this.data.ref())

  var ccTensor = this.refClone()
  ccTensor.override(etensor, this.dims)
  return ccTensor
};

// Torch does not support determinants, so we compute product of real eigenvalues
THTensor.prototype.determinant = function() {
  assert.ok(this.rank === 2);
  assert.ok(this.dims[0] === this.dims[1]);

  var etensor = THTensor.create_empty_of_size(this.data.ref());
  TH.THFloatTensor_geev(etensor.ref(), ref.NULL, this.ref, 'N');

  var evals = new THTensor([this.dims[0]]);
  TH.THFloatTensor_narrow(evals.ref, etensor.ref(), 0, 0, this.dims[0]);
  var ev = new THTensor(evals.dims);
  ev.slowCopy(evals);
  // ev = evals.clone();
  var det = TH.THFloatTensor_prodall(ev.ref);
  return det;
};

THTensor.prototype.dot = function(t) {
  var a = this, b = t;
//   console.log("Thtensor.dot:  ", b)
  console.log(this.dims)
  console.log(t.dims)
  if (a.rank !== 2 || b.rank !== 2)
    throw new Error('Inputs to dot should have rank = 2.');
  if (a.dims[1] !== b.dims[0])
    throw new Error('Dimension mismatch in dot. Inputs have dimension ' + a.dims + ' and ' + b.dims + '.');
//   var t_for_mul = TH.THFloatTensor_new().deref();
  var t_for_mul = new THTensor([a.dims[0], b.dims[1]]);
//   TH.THFloatTensor_resize2d(t_for_mul.ref, a.dims[0], b.dims[1]);
  var beta = 0, alpha = 1;
  TH.THFloatTensor_addmm(t_for_mul.data.ref(), beta, t_for_mul.data.ref(), alpha, a.data.ref(), b.data.ref());
  var mm_tensor = a.refClone();
  mm_tensor.override(t_for_mul.data, [a.dims[0], b.dims[1]]);
  return mm_tensor;
};

THTensor.prototype.cholesky = function() {
  assert.ok((this.rank === 2) && (this.dims[0] === this.dims[1]),
            'cholesky is only defined for square matrices.');

  var cc = THTensor.create_empty_of_size(this.data.ref())
  TH.THFloatTensor_potrf(this.data.ref(), cc.ref(), 'U');
  var ccTensor = this.refClone()
  ccTensor.override(cc, this.dims.slice(0))
  return ccTensor
};


module.exports = THTensor;

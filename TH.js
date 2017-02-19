var FFI = require('ffi'),
    ArrayType = require('ref-array'),
    Struct = require('ref-struct'),
    ref = require('ref');

var voidPtr = ref.refType(ref.types.void);

exports.CONSTANTS = {
};

var THErrorHandlerFunction = exports.THErrorHandlerFunction = FFI.Function(ref.types.void, [
  ref.types.CString,
  voidPtr,
]);
var THErrorHandlerFunctionPtr = exports.THErrorHandlerFunctionPtr = ref.refType(THErrorHandlerFunction);
var THArgErrorHandlerFunction = exports.THArgErrorHandlerFunction = FFI.Function(ref.types.void, [
  ref.types.int32,
  ref.types.CString,
  voidPtr,
]);
var THArgErrorHandlerFunctionPtr = exports.THArgErrorHandlerFunctionPtr = ref.refType(THArgErrorHandlerFunction);
var ptrdiff_t = exports.ptrdiff_t = voidPtr;
var ptrdiff_tPtr = exports.ptrdiff_tPtr = ref.refType(ptrdiff_t);
var THGenerator = exports.THGenerator = Struct({
  the_initial_seed: ref.types.ulong,
  left: ref.types.int32,
  seeded: ref.types.int32,
  next: ref.types.ulong,
  state: ArrayType(ref.types.ulong, 624),
  normal_x: ref.types.double,
  normal_y: ref.types.double,
  normal_rho: ref.types.double,
  normal_is_valid: ref.types.int32,
});
var THGeneratorPtr = exports.THGeneratorPtr = ref.refType(THGenerator);
var THMapAllocatorContext = exports.THMapAllocatorContext = voidPtr;
var THMapAllocatorContextPtr = exports.THMapAllocatorContextPtr = ref.refType(THMapAllocatorContext);
var THAllocator = exports.THAllocator = Struct({
  malloc: voidPtr,
  realloc: voidPtr,
  free: voidPtr,
});
var THAllocatorPtr = exports.THAllocatorPtr = ref.refType(THAllocator);
var THByteStorage = exports.THByteStorage = Struct({
  data: ref.refType(ref.types.uchar),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THByteStoragePtr = exports.THByteStoragePtr = ref.refType(THByteStorage);
var THCharStorage = exports.THCharStorage = Struct({
  data: ref.types.CString,
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THCharStoragePtr = exports.THCharStoragePtr = ref.refType(THCharStorage);
var THShortStorage = exports.THShortStorage = Struct({
  data: ref.refType(ref.types.short),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THShortStoragePtr = exports.THShortStoragePtr = ref.refType(THShortStorage);
var THIntStorage = exports.THIntStorage = Struct({
  data: ref.refType(ref.types.int32),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THIntStoragePtr = exports.THIntStoragePtr = ref.refType(THIntStorage);
var THLongStorage = exports.THLongStorage = Struct({
  data: ref.refType(ref.types.long),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THLongStoragePtr = exports.THLongStoragePtr = ref.refType(THLongStorage);
var THFloatStorage = exports.THFloatStorage = Struct({
  data: ref.refType(ref.types.float),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THFloatStoragePtr = exports.THFloatStoragePtr = ref.refType(THFloatStorage);
var THDoubleStorage = exports.THDoubleStorage = Struct({
  data: ref.refType(ref.types.double),
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THDoubleStoragePtr = exports.THDoubleStoragePtr = ref.refType(THDoubleStorage);
var THHalf = exports.THHalf = voidPtr;
var THHalfPtr = exports.THHalfPtr = ref.refType(THHalf);
var THHalfStorage = exports.THHalfStorage = Struct({
  data: THHalfPtr,
  size: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
  allocator: THAllocatorPtr,
  allocatorContext: voidPtr,
  view: voidPtr,
});
var THHalfStoragePtr = exports.THHalfStoragePtr = ref.refType(THHalfStorage);
var THByteTensor = exports.THByteTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THByteStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THByteTensorPtr = exports.THByteTensorPtr = ref.refType(THByteTensor);
var THDescBuff = exports.THDescBuff = Struct({
  str: ArrayType(ref.types.char, 64),
});
var THDescBuffPtr = exports.THDescBuffPtr = ref.refType(THDescBuff);
var THCharTensor = exports.THCharTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THCharStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THCharTensorPtr = exports.THCharTensorPtr = ref.refType(THCharTensor);
var THShortTensor = exports.THShortTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THShortStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THShortTensorPtr = exports.THShortTensorPtr = ref.refType(THShortTensor);
var THIntTensor = exports.THIntTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THIntStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THIntTensorPtr = exports.THIntTensorPtr = ref.refType(THIntTensor);
var THLongTensor = exports.THLongTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THLongStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THLongTensorPtr = exports.THLongTensorPtr = ref.refType(THLongTensor);
var THFloatTensor = exports.THFloatTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THFloatStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THFloatTensorPtr = exports.THFloatTensorPtr = ref.refType(THFloatTensor);
var THDoubleTensor = exports.THDoubleTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THDoubleStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THDoubleTensorPtr = exports.THDoubleTensorPtr = ref.refType(THDoubleTensor);
var THHalfTensor = exports.THHalfTensor = Struct({
  size: ref.refType(ref.types.long),
  stride: ref.refType(ref.types.long),
  nDimension: ref.types.int32,
  storage: THHalfStoragePtr,
  storageOffset: ref.types.long,
  refcount: ref.types.int32,
  flag: ref.types.char,
});
var THHalfTensorPtr = exports.THHalfTensorPtr = ref.refType(THHalfTensor);
var THFile = exports.THFile = voidPtr;
var THFilePtr = exports.THFilePtr = ref.refType(THFile);

exports.TH = new FFI.Library('libTH', {
  THLog1p: [ref.types.double, [
    ref.types.double,
  ]],
  THSetErrorHandler: [ref.types.void, [
    THErrorHandlerFunction,
    voidPtr,
  ]],
  THSetDefaultErrorHandler: [ref.types.void, [
    THErrorHandlerFunction,
    voidPtr,
  ]],
  THSetArgErrorHandler: [ref.types.void, [
    THArgErrorHandlerFunction,
    voidPtr,
  ]],
  THSetDefaultArgErrorHandler: [ref.types.void, [
    THArgErrorHandlerFunction,
    voidPtr,
  ]],
  THAlloc: [voidPtr, [
    ref.types.long,
  ]],
  THRealloc: [voidPtr, [
    voidPtr,
    ref.types.long,
  ]],
  THFree: [ref.types.void, [
    voidPtr,
  ]],
  THSetGCHandler: [ref.types.void, [
    voidPtr,
    voidPtr,
  ]],
  THHeapUpdate: [ref.types.void, [
    ref.types.long,
  ]],
  THSetNumThreads: [ref.types.void, [
    ref.types.int32,
  ]],
  THGetNumThreads: [ref.types.int32, [
  ]],
  THGetNumCores: [ref.types.int32, [
  ]],
  THByteBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_dot: [ref.types.uchar, [
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.refType(ref.types.uchar),
    ref.types.long,
    ref.types.uchar,
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THCharBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_dot: [ref.types.char, [
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
    ref.types.CString,
    ref.types.long,
    ref.types.char,
    ref.types.CString,
    ref.types.long,
  ]],
  THShortBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_dot: [ref.types.short, [
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.refType(ref.types.short),
    ref.types.long,
    ref.types.short,
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THIntBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_dot: [ref.types.int32, [
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.refType(ref.types.int32),
    ref.types.long,
    ref.types.int32,
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THLongBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_dot: [ref.types.long, [
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THFloatBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_dot: [ref.types.float, [
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THDoubleBlas_swap: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_scal: [ref.types.void, [
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_copy: [ref.types.void, [
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_axpy: [ref.types.void, [
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_dot: [ref.types.double, [
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_gemv: [ref.types.void, [
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_ger: [ref.types.void, [
    ref.types.long,
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleBlas_gemm: [ref.types.void, [
    ref.types.char,
    ref.types.char,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.refType(ref.types.double),
    ref.types.long,
    ref.types.double,
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THAtomicSet: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.types.int32,
  ]],
  THAtomicGet: [ref.types.int32, [
    ref.refType(ref.types.int32),
  ]],
  THAtomicAdd: [ref.types.int32, [
    ref.refType(ref.types.int32),
    ref.types.int32,
  ]],
  THAtomicCompareAndSwap: [ref.types.int32, [
    ref.refType(ref.types.int32),
    ref.types.int32,
    ref.types.int32,
  ]],
  THAtomicIncrementRef: [ref.types.void, [
    ref.refType(ref.types.int32),
  ]],
  THAtomicDecrementRef: [ref.types.int32, [
    ref.refType(ref.types.int32),
  ]],
  THAtomicSetLong: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THAtomicGetLong: [ref.types.long, [
    ref.refType(ref.types.long),
  ]],
  THAtomicAddLong: [ref.types.long, [
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THAtomicCompareAndSwapLong: [ref.types.long, [
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
  ]],
  THAtomicSetPtrdiff: [ref.types.void, [
    ptrdiff_t,
    ref.types.long,
  ]],
  THAtomicGetPtrdiff: [ref.types.long, [
    ptrdiff_tPtr,
  ]],
  THAtomicAddPtrdiff: [ref.types.long, [
    ptrdiff_tPtr,
    ref.types.long,
  ]],
  THAtomicCompareAndSwapPtrdiff: [ref.types.long, [
    ptrdiff_tPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THByteVector_fill: [ref.types.void, [
    ref.refType(ref.types.uchar),
    ref.types.uchar,
    ref.types.long,
  ]],
  THByteVector_add: [ref.types.void, [
    ref.refType(ref.types.uchar),
    ref.refType(ref.types.uchar),
    ref.types.uchar,
    ref.types.long,
  ]],
  THByteVector_diff: [ref.types.void, [
    ref.refType(ref.types.uchar),
    ref.refType(ref.types.uchar),
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteVector_scale: [ref.types.void, [
    ref.refType(ref.types.uchar),
    ref.types.uchar,
    ref.types.long,
  ]],
  THByteVector_mul: [ref.types.void, [
    ref.refType(ref.types.uchar),
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THCharVector_fill: [ref.types.void, [
    ref.types.CString,
    ref.types.char,
    ref.types.long,
  ]],
  THCharVector_add: [ref.types.void, [
    ref.types.CString,
    ref.types.CString,
    ref.types.char,
    ref.types.long,
  ]],
  THCharVector_diff: [ref.types.void, [
    ref.types.CString,
    ref.types.CString,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharVector_scale: [ref.types.void, [
    ref.types.CString,
    ref.types.char,
    ref.types.long,
  ]],
  THCharVector_mul: [ref.types.void, [
    ref.types.CString,
    ref.types.CString,
    ref.types.long,
  ]],
  THCharVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THShortVector_fill: [ref.types.void, [
    ref.refType(ref.types.short),
    ref.types.short,
    ref.types.long,
  ]],
  THShortVector_add: [ref.types.void, [
    ref.refType(ref.types.short),
    ref.refType(ref.types.short),
    ref.types.short,
    ref.types.long,
  ]],
  THShortVector_diff: [ref.types.void, [
    ref.refType(ref.types.short),
    ref.refType(ref.types.short),
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortVector_scale: [ref.types.void, [
    ref.refType(ref.types.short),
    ref.types.short,
    ref.types.long,
  ]],
  THShortVector_mul: [ref.types.void, [
    ref.refType(ref.types.short),
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THIntVector_fill: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.types.int32,
    ref.types.long,
  ]],
  THIntVector_add: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.refType(ref.types.int32),
    ref.types.int32,
    ref.types.long,
  ]],
  THIntVector_diff: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.refType(ref.types.int32),
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntVector_scale: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.types.int32,
    ref.types.long,
  ]],
  THIntVector_mul: [ref.types.void, [
    ref.refType(ref.types.int32),
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THLongVector_fill: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
  ]],
  THLongVector_add: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
  ]],
  THLongVector_diff: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.refType(ref.types.long),
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongVector_scale: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.types.long,
    ref.types.long,
  ]],
  THLongVector_mul: [ref.types.void, [
    ref.refType(ref.types.long),
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THFloatVector_fill: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.types.long,
  ]],
  THFloatVector_add: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.refType(ref.types.float),
    ref.types.float,
    ref.types.long,
  ]],
  THFloatVector_diff: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.refType(ref.types.float),
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatVector_scale: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.types.long,
  ]],
  THFloatVector_mul: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THDoubleVector_fill: [ref.types.void, [
    ref.refType(ref.types.double),
    ref.types.double,
    ref.types.long,
  ]],
  THDoubleVector_add: [ref.types.void, [
    ref.refType(ref.types.double),
    ref.refType(ref.types.double),
    ref.types.double,
    ref.types.long,
  ]],
  THDoubleVector_diff: [ref.types.void, [
    ref.refType(ref.types.double),
    ref.refType(ref.types.double),
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleVector_scale: [ref.types.void, [
    ref.refType(ref.types.double),
    ref.types.double,
    ref.types.long,
  ]],
  THDoubleVector_mul: [ref.types.void, [
    ref.refType(ref.types.double),
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleVector_vectorDispatchInit: [ref.types.void, [
  ]],
  THLogAdd: [ref.types.double, [
    ref.types.double,
    ref.types.double,
  ]],
  THLogSub: [ref.types.double, [
    ref.types.double,
    ref.types.double,
  ]],
  THExpMinusApprox: [ref.types.double, [
    ref.types.double,
  ]],
  THGenerator_new: [THGeneratorPtr, [
  ]],
  THGenerator_copy: [THGeneratorPtr, [
    THGeneratorPtr,
    THGeneratorPtr,
  ]],
  THGenerator_free: [ref.types.void, [
    THGeneratorPtr,
  ]],
  THGenerator_isValid: [ref.types.int32, [
    THGeneratorPtr,
  ]],
  THRandom_seed: [ref.types.ulong, [
    THGeneratorPtr,
  ]],
  THRandom_manualSeed: [ref.types.void, [
    THGeneratorPtr,
    ref.types.ulong,
  ]],
  THRandom_initialSeed: [ref.types.ulong, [
    THGeneratorPtr,
  ]],
  THRandom_random: [ref.types.ulong, [
    THGeneratorPtr,
  ]],
  THRandom_uniform: [ref.types.double, [
    THGeneratorPtr,
    ref.types.double,
    ref.types.double,
  ]],
  THRandom_normal: [ref.types.double, [
    THGeneratorPtr,
    ref.types.double,
    ref.types.double,
  ]],
  THRandom_exponential: [ref.types.double, [
    THGeneratorPtr,
    ref.types.double,
  ]],
  THRandom_cauchy: [ref.types.double, [
    THGeneratorPtr,
    ref.types.double,
    ref.types.double,
  ]],
  THRandom_logNormal: [ref.types.double, [
    THGeneratorPtr,
    ref.types.double,
    ref.types.double,
  ]],
  THRandom_geometric: [ref.types.int32, [
    THGeneratorPtr,
    ref.types.double,
  ]],
  THRandom_bernoulli: [ref.types.int32, [
    THGeneratorPtr,
    ref.types.double,
  ]],
  THMapAllocatorContext_new: [THMapAllocatorContext, [
    ref.types.CString,
    ref.types.int32,
  ]],
  THMapAllocatorContext_newWithFd: [THMapAllocatorContextPtr, [
    ref.types.CString,
    ref.types.int32,
    ref.types.int32,
  ]],
  THMapAllocatorContext_filename: [ref.types.CString, [
    THMapAllocatorContextPtr,
  ]],
  THMapAllocatorContext_fd: [ref.types.int32, [
    THMapAllocatorContextPtr,
  ]],
  THMapAllocatorContext_size: [ref.types.long, [
    THMapAllocatorContextPtr,
  ]],
  THMapAllocatorContext_free: [ref.types.void, [
    THMapAllocatorContextPtr,
  ]],
  THRefcountedMapAllocator_incref: [ref.types.void, [
    THMapAllocatorContextPtr,
    voidPtr,
  ]],
  THRefcountedMapAllocator_decref: [ref.types.int32, [
    THMapAllocatorContextPtr,
    voidPtr,
  ]],
  THByteStorage_data: [ref.refType(ref.types.uchar), [
    THByteStoragePtr,
  ]],
  THByteStorage_size: [ref.types.long, [
    THByteStoragePtr,
  ]],
  THByteStorage_elementSize: [ref.types.ulong, [
  ]],
  THByteStorage_set: [ref.types.void, [
    THByteStoragePtr,
    ref.types.long,
    ref.types.uchar,
  ]],
  THByteStorage_get: [ref.types.uchar, [
    THByteStoragePtr,
    ref.types.long,
  ]],
  THByteStorage_new: [THByteStoragePtr, [
  ]],
  THByteStorage_newWithSize: [THByteStoragePtr, [
    ref.types.long,
  ]],
  THByteStorage_newWithSize1: [THByteStoragePtr, [
    ref.types.uchar,
  ]],
  THByteStorage_newWithSize2: [THByteStoragePtr, [
    ref.types.uchar,
    ref.types.uchar,
  ]],
  THByteStorage_newWithSize3: [THByteStoragePtr, [
    ref.types.uchar,
    ref.types.uchar,
    ref.types.uchar,
  ]],
  THByteStorage_newWithSize4: [THByteStoragePtr, [
    ref.types.uchar,
    ref.types.uchar,
    ref.types.uchar,
    ref.types.uchar,
  ]],
  THByteStorage_newWithMapping: [THByteStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THByteStorage_newWithData: [THByteStoragePtr, [
    ref.refType(ref.types.uchar),
    ref.types.long,
  ]],
  THByteStorage_newWithAllocator: [THByteStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THByteStorage_newWithDataAndAllocator: [THByteStoragePtr, [
    ref.refType(ref.types.uchar),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THByteStorage_setFlag: [ref.types.void, [
    THByteStoragePtr,
    ref.types.char,
  ]],
  THByteStorage_clearFlag: [ref.types.void, [
    THByteStoragePtr,
    ref.types.char,
  ]],
  THByteStorage_retain: [ref.types.void, [
    THByteStoragePtr,
  ]],
  THByteStorage_swap: [ref.types.void, [
    THByteStoragePtr,
    THByteStoragePtr,
  ]],
  THByteStorage_free: [ref.types.void, [
    THByteStoragePtr,
  ]],
  THByteStorage_resize: [ref.types.void, [
    THByteStoragePtr,
    ref.types.long,
  ]],
  THByteStorage_fill: [ref.types.void, [
    THByteStoragePtr,
    ref.types.uchar,
  ]],
  THCharStorage_data: [ref.types.CString, [
    THCharStoragePtr,
  ]],
  THCharStorage_size: [ref.types.long, [
    THCharStoragePtr,
  ]],
  THCharStorage_elementSize: [ref.types.ulong, [
  ]],
  THCharStorage_set: [ref.types.void, [
    THCharStoragePtr,
    ref.types.long,
    ref.types.char,
  ]],
  THCharStorage_get: [ref.types.char, [
    THCharStoragePtr,
    ref.types.long,
  ]],
  THCharStorage_new: [THCharStoragePtr, [
  ]],
  THCharStorage_newWithSize: [THCharStoragePtr, [
    ref.types.long,
  ]],
  THCharStorage_newWithSize1: [THCharStoragePtr, [
    ref.types.char,
  ]],
  THCharStorage_newWithSize2: [THCharStoragePtr, [
    ref.types.char,
    ref.types.char,
  ]],
  THCharStorage_newWithSize3: [THCharStoragePtr, [
    ref.types.char,
    ref.types.char,
    ref.types.char,
  ]],
  THCharStorage_newWithSize4: [THCharStoragePtr, [
    ref.types.char,
    ref.types.char,
    ref.types.char,
    ref.types.char,
  ]],
  THCharStorage_newWithMapping: [THCharStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THCharStorage_newWithData: [THCharStoragePtr, [
    ref.types.CString,
    ref.types.long,
  ]],
  THCharStorage_newWithAllocator: [THCharStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THCharStorage_newWithDataAndAllocator: [THCharStoragePtr, [
    ref.types.CString,
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THCharStorage_setFlag: [ref.types.void, [
    THCharStoragePtr,
    ref.types.char,
  ]],
  THCharStorage_clearFlag: [ref.types.void, [
    THCharStoragePtr,
    ref.types.char,
  ]],
  THCharStorage_retain: [ref.types.void, [
    THCharStoragePtr,
  ]],
  THCharStorage_swap: [ref.types.void, [
    THCharStoragePtr,
    THCharStoragePtr,
  ]],
  THCharStorage_free: [ref.types.void, [
    THCharStoragePtr,
  ]],
  THCharStorage_resize: [ref.types.void, [
    THCharStoragePtr,
    ref.types.long,
  ]],
  THCharStorage_fill: [ref.types.void, [
    THCharStoragePtr,
    ref.types.char,
  ]],
  THShortStorage_data: [ref.refType(ref.types.short), [
    THShortStoragePtr,
  ]],
  THShortStorage_size: [ref.types.long, [
    THShortStoragePtr,
  ]],
  THShortStorage_elementSize: [ref.types.ulong, [
  ]],
  THShortStorage_set: [ref.types.void, [
    THShortStoragePtr,
    ref.types.long,
    ref.types.short,
  ]],
  THShortStorage_get: [ref.types.short, [
    THShortStoragePtr,
    ref.types.long,
  ]],
  THShortStorage_new: [THShortStoragePtr, [
  ]],
  THShortStorage_newWithSize: [THShortStoragePtr, [
    ref.types.long,
  ]],
  THShortStorage_newWithSize1: [THShortStoragePtr, [
    ref.types.short,
  ]],
  THShortStorage_newWithSize2: [THShortStoragePtr, [
    ref.types.short,
    ref.types.short,
  ]],
  THShortStorage_newWithSize3: [THShortStoragePtr, [
    ref.types.short,
    ref.types.short,
    ref.types.short,
  ]],
  THShortStorage_newWithSize4: [THShortStoragePtr, [
    ref.types.short,
    ref.types.short,
    ref.types.short,
    ref.types.short,
  ]],
  THShortStorage_newWithMapping: [THShortStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THShortStorage_newWithData: [THShortStoragePtr, [
    ref.refType(ref.types.short),
    ref.types.long,
  ]],
  THShortStorage_newWithAllocator: [THShortStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THShortStorage_newWithDataAndAllocator: [THShortStoragePtr, [
    ref.refType(ref.types.short),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THShortStorage_setFlag: [ref.types.void, [
    THShortStoragePtr,
    ref.types.char,
  ]],
  THShortStorage_clearFlag: [ref.types.void, [
    THShortStoragePtr,
    ref.types.char,
  ]],
  THShortStorage_retain: [ref.types.void, [
    THShortStoragePtr,
  ]],
  THShortStorage_swap: [ref.types.void, [
    THShortStoragePtr,
    THShortStoragePtr,
  ]],
  THShortStorage_free: [ref.types.void, [
    THShortStoragePtr,
  ]],
  THShortStorage_resize: [ref.types.void, [
    THShortStoragePtr,
    ref.types.long,
  ]],
  THShortStorage_fill: [ref.types.void, [
    THShortStoragePtr,
    ref.types.short,
  ]],
  THIntStorage_data: [ref.refType(ref.types.int32), [
    THIntStoragePtr,
  ]],
  THIntStorage_size: [ref.types.long, [
    THIntStoragePtr,
  ]],
  THIntStorage_elementSize: [ref.types.ulong, [
  ]],
  THIntStorage_set: [ref.types.void, [
    THIntStoragePtr,
    ref.types.long,
    ref.types.int32,
  ]],
  THIntStorage_get: [ref.types.int32, [
    THIntStoragePtr,
    ref.types.long,
  ]],
  THIntStorage_new: [THIntStoragePtr, [
  ]],
  THIntStorage_newWithSize: [THIntStoragePtr, [
    ref.types.long,
  ]],
  THIntStorage_newWithSize1: [THIntStoragePtr, [
    ref.types.int32,
  ]],
  THIntStorage_newWithSize2: [THIntStoragePtr, [
    ref.types.int32,
    ref.types.int32,
  ]],
  THIntStorage_newWithSize3: [THIntStoragePtr, [
    ref.types.int32,
    ref.types.int32,
    ref.types.int32,
  ]],
  THIntStorage_newWithSize4: [THIntStoragePtr, [
    ref.types.int32,
    ref.types.int32,
    ref.types.int32,
    ref.types.int32,
  ]],
  THIntStorage_newWithMapping: [THIntStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THIntStorage_newWithData: [THIntStoragePtr, [
    ref.refType(ref.types.int32),
    ref.types.long,
  ]],
  THIntStorage_newWithAllocator: [THIntStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THIntStorage_newWithDataAndAllocator: [THIntStoragePtr, [
    ref.refType(ref.types.int32),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THIntStorage_setFlag: [ref.types.void, [
    THIntStoragePtr,
    ref.types.char,
  ]],
  THIntStorage_clearFlag: [ref.types.void, [
    THIntStoragePtr,
    ref.types.char,
  ]],
  THIntStorage_retain: [ref.types.void, [
    THIntStoragePtr,
  ]],
  THIntStorage_swap: [ref.types.void, [
    THIntStoragePtr,
    THIntStoragePtr,
  ]],
  THIntStorage_free: [ref.types.void, [
    THIntStoragePtr,
  ]],
  THIntStorage_resize: [ref.types.void, [
    THIntStoragePtr,
    ref.types.long,
  ]],
  THIntStorage_fill: [ref.types.void, [
    THIntStoragePtr,
    ref.types.int32,
  ]],
  THLongStorage_data: [ref.refType(ref.types.long), [
    THLongStoragePtr,
  ]],
  THLongStorage_size: [ref.types.long, [
    THLongStoragePtr,
  ]],
  THLongStorage_elementSize: [ref.types.ulong, [
  ]],
  THLongStorage_set: [ref.types.void, [
    THLongStoragePtr,
    ref.types.long,
    ref.types.long,
  ]],
  THLongStorage_get: [ref.types.long, [
    THLongStoragePtr,
    ref.types.long,
  ]],
  THLongStorage_new: [THLongStoragePtr, [
  ]],
  THLongStorage_newWithSize: [THLongStoragePtr, [
    ref.types.long,
  ]],
  THLongStorage_newWithSize1: [THLongStoragePtr, [
    ref.types.long,
  ]],
  THLongStorage_newWithSize2: [THLongStoragePtr, [
    ref.types.long,
    ref.types.long,
  ]],
  THLongStorage_newWithSize3: [THLongStoragePtr, [
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THLongStorage_newWithSize4: [THLongStoragePtr, [
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THLongStorage_newWithMapping: [THLongStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THLongStorage_newWithData: [THLongStoragePtr, [
    ref.refType(ref.types.long),
    ref.types.long,
  ]],
  THLongStorage_newWithAllocator: [THLongStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THLongStorage_newWithDataAndAllocator: [THLongStoragePtr, [
    ref.refType(ref.types.long),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THLongStorage_setFlag: [ref.types.void, [
    THLongStoragePtr,
    ref.types.char,
  ]],
  THLongStorage_clearFlag: [ref.types.void, [
    THLongStoragePtr,
    ref.types.char,
  ]],
  THLongStorage_retain: [ref.types.void, [
    THLongStoragePtr,
  ]],
  THLongStorage_swap: [ref.types.void, [
    THLongStoragePtr,
    THLongStoragePtr,
  ]],
  THLongStorage_free: [ref.types.void, [
    THLongStoragePtr,
  ]],
  THLongStorage_resize: [ref.types.void, [
    THLongStoragePtr,
    ref.types.long,
  ]],
  THLongStorage_fill: [ref.types.void, [
    THLongStoragePtr,
    ref.types.long,
  ]],
  THFloatStorage_data: [ref.refType(ref.types.float), [
    THFloatStoragePtr,
  ]],
  THFloatStorage_size: [ref.types.long, [
    THFloatStoragePtr,
  ]],
  THFloatStorage_elementSize: [ref.types.ulong, [
  ]],
  THFloatStorage_set: [ref.types.void, [
    THFloatStoragePtr,
    ref.types.long,
    ref.types.float,
  ]],
  THFloatStorage_get: [ref.types.float, [
    THFloatStoragePtr,
    ref.types.long,
  ]],
  THFloatStorage_new: [THFloatStoragePtr, [
  ]],
  THFloatStorage_newWithSize: [THFloatStoragePtr, [
    ref.types.long,
  ]],
  THFloatStorage_newWithSize1: [THFloatStoragePtr, [
    ref.types.float,
  ]],
  THFloatStorage_newWithSize2: [THFloatStoragePtr, [
    ref.types.float,
    ref.types.float,
  ]],
  THFloatStorage_newWithSize3: [THFloatStoragePtr, [
    ref.types.float,
    ref.types.float,
    ref.types.float,
  ]],
  THFloatStorage_newWithSize4: [THFloatStoragePtr, [
    ref.types.float,
    ref.types.float,
    ref.types.float,
    ref.types.float,
  ]],
  THFloatStorage_newWithMapping: [THFloatStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THFloatStorage_newWithData: [THFloatStoragePtr, [
    ref.refType(ref.types.float),
    ref.types.long,
  ]],
  THFloatStorage_newWithAllocator: [THFloatStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THFloatStorage_newWithDataAndAllocator: [THFloatStoragePtr, [
    ref.refType(ref.types.float),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THFloatStorage_setFlag: [ref.types.void, [
    THFloatStoragePtr,
    ref.types.char,
  ]],
  THFloatStorage_clearFlag: [ref.types.void, [
    THFloatStoragePtr,
    ref.types.char,
  ]],
  THFloatStorage_retain: [ref.types.void, [
    THFloatStoragePtr,
  ]],
  THFloatStorage_swap: [ref.types.void, [
    THFloatStoragePtr,
    THFloatStoragePtr,
  ]],
  THFloatStorage_free: [ref.types.void, [
    THFloatStoragePtr,
  ]],
  THFloatStorage_resize: [ref.types.void, [
    THFloatStoragePtr,
    ref.types.long,
  ]],
  THFloatStorage_fill: [ref.types.void, [
    THFloatStoragePtr,
    ref.types.float,
  ]],
  THDoubleStorage_data: [ref.refType(ref.types.double), [
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_size: [ref.types.long, [
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_elementSize: [ref.types.ulong, [
  ]],
  THDoubleStorage_set: [ref.types.void, [
    THDoubleStoragePtr,
    ref.types.long,
    ref.types.double,
  ]],
  THDoubleStorage_get: [ref.types.double, [
    THDoubleStoragePtr,
    ref.types.long,
  ]],
  THDoubleStorage_new: [THDoubleStoragePtr, [
  ]],
  THDoubleStorage_newWithSize: [THDoubleStoragePtr, [
    ref.types.long,
  ]],
  THDoubleStorage_newWithSize1: [THDoubleStoragePtr, [
    ref.types.double,
  ]],
  THDoubleStorage_newWithSize2: [THDoubleStoragePtr, [
    ref.types.double,
    ref.types.double,
  ]],
  THDoubleStorage_newWithSize3: [THDoubleStoragePtr, [
    ref.types.double,
    ref.types.double,
    ref.types.double,
  ]],
  THDoubleStorage_newWithSize4: [THDoubleStoragePtr, [
    ref.types.double,
    ref.types.double,
    ref.types.double,
    ref.types.double,
  ]],
  THDoubleStorage_newWithMapping: [THDoubleStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THDoubleStorage_newWithData: [THDoubleStoragePtr, [
    ref.refType(ref.types.double),
    ref.types.long,
  ]],
  THDoubleStorage_newWithAllocator: [THDoubleStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THDoubleStorage_newWithDataAndAllocator: [THDoubleStoragePtr, [
    ref.refType(ref.types.double),
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THDoubleStorage_setFlag: [ref.types.void, [
    THDoubleStoragePtr,
    ref.types.char,
  ]],
  THDoubleStorage_clearFlag: [ref.types.void, [
    THDoubleStoragePtr,
    ref.types.char,
  ]],
  THDoubleStorage_retain: [ref.types.void, [
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_swap: [ref.types.void, [
    THDoubleStoragePtr,
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_free: [ref.types.void, [
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_resize: [ref.types.void, [
    THDoubleStoragePtr,
    ref.types.long,
  ]],
  THDoubleStorage_fill: [ref.types.void, [
    THDoubleStoragePtr,
    ref.types.double,
  ]],
  THHalfStorage_data: [THHalf, [
    THHalfStoragePtr,
  ]],
  THHalfStorage_size: [ref.types.long, [
    THHalfStoragePtr,
  ]],
  THHalfStorage_elementSize: [ref.types.ulong, [
  ]],
  THHalfStorage_set: [ref.types.void, [
    THHalfStoragePtr,
    ref.types.long,
    THHalf,
  ]],
  THHalfStorage_get: [THHalf, [
    THHalfStoragePtr,
    ref.types.long,
  ]],
  THHalfStorage_new: [THHalfStoragePtr, [
  ]],
  THHalfStorage_newWithSize: [THHalfStoragePtr, [
    ref.types.long,
  ]],
  THHalfStorage_newWithSize1: [THHalfStoragePtr, [
    THHalf,
  ]],
  THHalfStorage_newWithSize2: [THHalfStoragePtr, [
    THHalf,
    THHalf,
  ]],
  THHalfStorage_newWithSize3: [THHalfStoragePtr, [
    THHalf,
    THHalf,
    THHalf,
  ]],
  THHalfStorage_newWithSize4: [THHalfStoragePtr, [
    THHalf,
    THHalf,
    THHalf,
    THHalf,
  ]],
  THHalfStorage_newWithMapping: [THHalfStoragePtr, [
    ref.types.CString,
    ref.types.long,
    ref.types.int32,
  ]],
  THHalfStorage_newWithData: [THHalfStoragePtr, [
    THHalfPtr,
    ref.types.long,
  ]],
  THHalfStorage_newWithAllocator: [THHalfStoragePtr, [
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THHalfStorage_newWithDataAndAllocator: [THHalfStoragePtr, [
    THHalfPtr,
    ref.types.long,
    THAllocatorPtr,
    voidPtr,
  ]],
  THHalfStorage_setFlag: [ref.types.void, [
    THHalfStoragePtr,
    ref.types.char,
  ]],
  THHalfStorage_clearFlag: [ref.types.void, [
    THHalfStoragePtr,
    ref.types.char,
  ]],
  THHalfStorage_retain: [ref.types.void, [
    THHalfStoragePtr,
  ]],
  THHalfStorage_swap: [ref.types.void, [
    THHalfStoragePtr,
    THHalfStoragePtr,
  ]],
  THHalfStorage_free: [ref.types.void, [
    THHalfStoragePtr,
  ]],
  THHalfStorage_resize: [ref.types.void, [
    THHalfStoragePtr,
    ref.types.long,
  ]],
  THHalfStorage_fill: [ref.types.void, [
    THHalfStoragePtr,
    THHalf,
  ]],
  THByteStorage_rawCopy: [ref.types.void, [
    THByteStoragePtr,
    ref.refType(ref.types.uchar),
  ]],
  THByteStorage_copy: [ref.types.void, [
    THByteStoragePtr,
    THByteStoragePtr,
  ]],
  THByteStorage_copyByte: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyChar: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyShort: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyInt: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyLong: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyFloat: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyDouble: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THByteStorage_copyHalf: [ref.types.void, [
    THByteStoragePtr,
    voidPtr,
  ]],
  THCharStorage_rawCopy: [ref.types.void, [
    THCharStoragePtr,
    ref.types.CString,
  ]],
  THCharStorage_copy: [ref.types.void, [
    THCharStoragePtr,
    THCharStoragePtr,
  ]],
  THCharStorage_copyByte: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyChar: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyShort: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyInt: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyLong: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyFloat: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyDouble: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THCharStorage_copyHalf: [ref.types.void, [
    THCharStoragePtr,
    voidPtr,
  ]],
  THShortStorage_rawCopy: [ref.types.void, [
    THShortStoragePtr,
    ref.refType(ref.types.short),
  ]],
  THShortStorage_copy: [ref.types.void, [
    THShortStoragePtr,
    THShortStoragePtr,
  ]],
  THShortStorage_copyByte: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyChar: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyShort: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyInt: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyLong: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyFloat: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyDouble: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THShortStorage_copyHalf: [ref.types.void, [
    THShortStoragePtr,
    voidPtr,
  ]],
  THIntStorage_rawCopy: [ref.types.void, [
    THIntStoragePtr,
    ref.refType(ref.types.int32),
  ]],
  THIntStorage_copy: [ref.types.void, [
    THIntStoragePtr,
    THIntStoragePtr,
  ]],
  THIntStorage_copyByte: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyChar: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyShort: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyInt: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyLong: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyFloat: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyDouble: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THIntStorage_copyHalf: [ref.types.void, [
    THIntStoragePtr,
    voidPtr,
  ]],
  THLongStorage_rawCopy: [ref.types.void, [
    THLongStoragePtr,
    ref.refType(ref.types.long),
  ]],
  THLongStorage_copy: [ref.types.void, [
    THLongStoragePtr,
    THLongStoragePtr,
  ]],
  THLongStorage_copyByte: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyChar: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyShort: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyInt: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyLong: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyFloat: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyDouble: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THLongStorage_copyHalf: [ref.types.void, [
    THLongStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_rawCopy: [ref.types.void, [
    THFloatStoragePtr,
    ref.refType(ref.types.float),
  ]],
  THFloatStorage_copy: [ref.types.void, [
    THFloatStoragePtr,
    THFloatStoragePtr,
  ]],
  THFloatStorage_copyByte: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyChar: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyShort: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyInt: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyLong: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyFloat: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyDouble: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THFloatStorage_copyHalf: [ref.types.void, [
    THFloatStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_rawCopy: [ref.types.void, [
    THDoubleStoragePtr,
    ref.refType(ref.types.double),
  ]],
  THDoubleStorage_copy: [ref.types.void, [
    THDoubleStoragePtr,
    THDoubleStoragePtr,
  ]],
  THDoubleStorage_copyByte: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyChar: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyShort: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyInt: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyLong: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyFloat: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyDouble: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THDoubleStorage_copyHalf: [ref.types.void, [
    THDoubleStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_rawCopy: [ref.types.void, [
    THHalfStoragePtr,
    THHalfPtr,
  ]],
  THHalfStorage_copy: [ref.types.void, [
    THHalfStoragePtr,
    THHalfStoragePtr,
  ]],
  THHalfStorage_copyByte: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyChar: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyShort: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyInt: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyLong: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyFloat: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyDouble: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THHalfStorage_copyHalf: [ref.types.void, [
    THHalfStoragePtr,
    voidPtr,
  ]],
  THFloatTensor_fill: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_zero: [ref.types.void, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_maskedFill: [ref.types.void, [
    THFloatTensorPtr,
    THByteTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_maskedCopy: [ref.types.void, [
    THFloatTensorPtr,
    THByteTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_maskedSelect: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THByteTensorPtr,
  ]],
  THFloatTensor_nonzero: [ref.types.void, [
    THLongTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_indexSelect: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
  ]],
  THFloatTensor_indexCopy: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_indexAdd: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_indexFill: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_gather: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
  ]],
  THFloatTensor_scatter: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_scatterFill: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.int32,
    THLongTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_dot: [ref.types.double, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_minall: [ref.types.float, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_maxall: [ref.types.float, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_sumall: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_prodall: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_neg: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cinv: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_add: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_sub: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_mul: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_div: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_fmod: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_remainder: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_clamp: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
  ]],
  THFloatTensor_cadd: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
  ]],
  THFloatTensor_csub: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cmul: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cpow: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cdiv: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cfmod: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cremainder: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addcmul: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addcdiv: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addmv: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addmm: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addr: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_addbmm: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_baddbmm: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_match: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_numel: [ref.types.long, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_max: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_min: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_kthvalue: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.int32,
  ]],
  THFloatTensor_mode: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_median: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_sum: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_prod: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_cumsum: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_cumprod: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_sign: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_trace: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_cross: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_cmax: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cmin: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cmaxValue: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_cminValue: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_zeros: [ref.types.void, [
    THFloatTensorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_ones: [ref.types.void, [
    THFloatTensorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_diag: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_eye: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_range: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.double,
    ref.types.double,
    ref.types.double,
  ]],
  THFloatTensor_randperm: [ref.types.void, [
    THFloatTensorPtr,
    THGeneratorPtr,
    ref.types.long,
  ]],
  THFloatTensor_reshape: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_sort: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_topk: [ref.types.void, [
    THFloatTensorPtr,
    THLongTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.int32,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_tril: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
  ]],
  THFloatTensor_triu: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
  ]],
  THFloatTensor_cat: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_catArray: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_equal: [ref.types.int32, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_ltValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_leValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_gtValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_geValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_neValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_eqValue: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_ltValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_leValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_gtValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_geValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_neValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_eqValueT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_ltTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_leTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_gtTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_geTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_neTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_eqTensor: [ref.types.void, [
    THByteTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_ltTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_leTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_gtTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_geTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_neTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_eqTensorT: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_sigmoid: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_log: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_log1p: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_exp: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cos: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_acos: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_cosh: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_sin: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_asin: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_sinh: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_tan: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_atan: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_atan2: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_tanh: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_pow: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_tpow: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    THFloatTensorPtr,
  ]],
  THFloatTensor_sqrt: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_rsqrt: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_ceil: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_floor: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_round: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_abs: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_trunc: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_frac: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_lerp: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_mean: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_std: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_var: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_norm: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    ref.types.int32,
  ]],
  THFloatTensor_renorm: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
    ref.types.int32,
    ref.types.float,
  ]],
  THFloatTensor_dist: [ref.types.double, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_histc: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.float,
    ref.types.float,
  ]],
  THFloatTensor_bhistc: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.float,
    ref.types.float,
  ]],
  THFloatTensor_meanall: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_varall: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_stdall: [ref.types.double, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_normall: [ref.types.double, [
    THFloatTensorPtr,
    ref.types.float,
  ]],
  THFloatTensor_linspace: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    ref.types.long,
  ]],
  THFloatTensor_logspace: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    ref.types.long,
  ]],
  THFloatTensor_rand: [ref.types.void, [
    THFloatTensorPtr,
    THGeneratorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_randn: [ref.types.void, [
    THFloatTensorPtr,
    THGeneratorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_copy: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_copyByte: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_copyChar: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_copyShort: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_copyInt: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_copyLong: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_copyFloat: [ref.types.void, [
    THFloatTensorPtr,
    voidPtr,
  ]],
  THFloatTensor_conv2DRevger: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_conv2DRevgerm: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_conv2Dger: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv2Dmv: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv2Dmm: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv2Dmul: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv2Dcmul: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_validXCorr3Dptr: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_validConv3Dptr: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_fullXCorr3Dptr: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_fullConv3Dptr: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_validXCorr3DRevptr: [ref.types.void, [
    ref.refType(ref.types.float),
    ref.types.float,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.refType(ref.types.float),
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_conv3DRevger: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_conv3Dger: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv3Dmv: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv3Dmul: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_conv3Dcmul: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.float,
    ref.types.float,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_gesv: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_trtrs: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_gels: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_syev: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_geev: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_gesvd: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_gesvd2: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_getri: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_potrf: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_potrs: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_potri: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
  ]],
  THFloatTensor_qr: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_geqrf: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_orgqr: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_ormqr: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
    ref.types.CString,
  ]],
  THFloatTensor_pstrf: [ref.types.void, [
    THFloatTensorPtr,
    THIntTensorPtr,
    THFloatTensorPtr,
    ref.types.CString,
    ref.types.float,
  ]],
  THFloatTensor_newWithSize: [THFloatTensorPtr, [
    THLongStoragePtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_newWithSize1d: [THFloatTensorPtr, [
    ref.types.long,
  ]],
  THFloatTensor_newWithSize2d: [THFloatTensorPtr, [
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_newWithSize3d: [THFloatTensorPtr, [
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_newWithSize4d: [THFloatTensorPtr, [
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_newClone: [THFloatTensorPtr, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_newContiguous: [THFloatTensorPtr, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_newSelect: [THFloatTensorPtr, [
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
  ]],
  THFloatTensor_newNarrow: [THFloatTensorPtr, [
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_newTranspose: [THFloatTensorPtr, [
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_newUnfold: [THFloatTensorPtr, [
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_resize: [ref.types.void, [
    THFloatTensorPtr,
    THLongStoragePtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_resizeAs: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_resize1d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
  ]],
  THFloatTensor_resize2d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_resize3d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_resize4d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_resize5d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_set: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_setStorage: [ref.types.void, [
    THFloatTensorPtr,
    THFloatStoragePtr,
    ref.types.long,
    THLongStoragePtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_setStorage1d: [ref.types.void, [
    THFloatTensorPtr,
    THFloatStoragePtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_setStorage2d: [ref.types.void, [
    THFloatTensorPtr,
    THFloatStoragePtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_setStorage3d: [ref.types.void, [
    THFloatTensorPtr,
    THFloatStoragePtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_setStorage4d: [ref.types.void, [
    THFloatTensorPtr,
    THFloatStoragePtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_narrow: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_select: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
  ]],
  THFloatTensor_transpose: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.int32,
  ]],
  THFloatTensor_unfold: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_squeeze: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_squeeze1d: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_isContiguous: [ref.types.int32, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_isSameSizeAs: [ref.types.int32, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_isSetTo: [ref.types.int32, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_isSize: [ref.types.int32, [
    THFloatTensorPtr,
    THLongStoragePtr,
  ]],
  THFloatTensor_nElement: [ref.types.long, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_retain: [ref.types.void, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_free: [ref.types.void, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_freeCopyTo: [ref.types.void, [
    THFloatTensorPtr,
    THFloatTensorPtr,
  ]],
  THFloatTensor_set1d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.float,
  ]],
  THFloatTensor_set2d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.float,
  ]],
  THFloatTensor_set3d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.float,
  ]],
  THFloatTensor_set4d: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.float,
  ]],
  THFloatTensor_get1d: [ref.types.float, [
    THFloatTensorPtr,
    ref.types.long,
  ]],
  THFloatTensor_get2d: [ref.types.float, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_get3d: [ref.types.float, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_get4d: [ref.types.float, [
    THFloatTensorPtr,
    ref.types.long,
    ref.types.long,
    ref.types.long,
    ref.types.long,
  ]],
  THFloatTensor_desc: [THDescBuff, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_storage: [THFloatStoragePtr, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_storageOffset: [ref.types.long, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_nDimension: [ref.types.int32, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_sizeDesc: [THDescBuff, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_size: [ref.types.long, [
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_stride: [ref.types.long, [
    THFloatTensorPtr,
    ref.types.int32,
  ]],
  THFloatTensor_newSizeOf: [THLongStoragePtr, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_newStrideOf: [THLongStoragePtr, [
    THFloatTensorPtr,
  ]],
  THFloatTensor_data: [ref.refType(ref.types.float), [
    THFloatTensorPtr,
  ]],
  THFloatTensor_setFlag: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.char,
  ]],
  THFloatTensor_clearFlag: [ref.types.void, [
    THFloatTensorPtr,
    ref.types.char,
  ]],
  THFloatTensor_new: [THFloatTensorPtr, [
  ]],
  THFloatTensor_newWithTensor: [THFloatTensorPtr, [
    THFloatTensorPtr,
  ]],
  THFile_isOpened: [ref.types.int32, [
    THFile,
  ]],
  THFile_isQuiet: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_isReadable: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_isWritable: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_isBinary: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_isAutoSpacing: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_hasError: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_binary: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_ascii: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_autoSpacing: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_noAutoSpacing: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_quiet: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_pedantic: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_clearError: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_readByteScalar: [ref.types.uchar, [
    THFilePtr,
  ]],
  THFile_readCharScalar: [ref.types.char, [
    THFilePtr,
  ]],
  THFile_readShortScalar: [ref.types.short, [
    THFilePtr,
  ]],
  THFile_readIntScalar: [ref.types.int32, [
    THFilePtr,
  ]],
  THFile_readLongScalar: [ref.types.long, [
    THFilePtr,
  ]],
  THFile_readFloatScalar: [ref.types.float, [
    THFilePtr,
  ]],
  THFile_readDoubleScalar: [ref.types.double, [
    THFilePtr,
  ]],
  THFile_writeByteScalar: [ref.types.void, [
    THFilePtr,
    ref.types.uchar,
  ]],
  THFile_writeCharScalar: [ref.types.void, [
    THFilePtr,
    ref.types.char,
  ]],
  THFile_writeShortScalar: [ref.types.void, [
    THFilePtr,
    ref.types.short,
  ]],
  THFile_writeIntScalar: [ref.types.void, [
    THFilePtr,
    ref.types.int32,
  ]],
  THFile_writeLongScalar: [ref.types.void, [
    THFilePtr,
    ref.types.long,
  ]],
  THFile_writeFloatScalar: [ref.types.void, [
    THFilePtr,
    ref.types.float,
  ]],
  THFile_writeDoubleScalar: [ref.types.void, [
    THFilePtr,
    ref.types.double,
  ]],
  THFile_readByte: [ref.types.ulong, [
    THFilePtr,
    THByteStoragePtr,
  ]],
  THFile_readChar: [ref.types.ulong, [
    THFilePtr,
    THCharStoragePtr,
  ]],
  THFile_readShort: [ref.types.ulong, [
    THFilePtr,
    THShortStoragePtr,
  ]],
  THFile_readInt: [ref.types.ulong, [
    THFilePtr,
    THIntStoragePtr,
  ]],
  THFile_readLong: [ref.types.ulong, [
    THFilePtr,
    THLongStoragePtr,
  ]],
  THFile_readFloat: [ref.types.ulong, [
    THFilePtr,
    THFloatStoragePtr,
  ]],
  THFile_readDouble: [ref.types.ulong, [
    THFilePtr,
    THDoubleStoragePtr,
  ]],
  THFile_writeByte: [ref.types.ulong, [
    THFilePtr,
    THByteStoragePtr,
  ]],
  THFile_writeChar: [ref.types.ulong, [
    THFilePtr,
    THCharStoragePtr,
  ]],
  THFile_writeShort: [ref.types.ulong, [
    THFilePtr,
    THShortStoragePtr,
  ]],
  THFile_writeInt: [ref.types.ulong, [
    THFilePtr,
    THIntStoragePtr,
  ]],
  THFile_writeLong: [ref.types.ulong, [
    THFilePtr,
    THLongStoragePtr,
  ]],
  THFile_writeFloat: [ref.types.ulong, [
    THFilePtr,
    THFloatStoragePtr,
  ]],
  THFile_writeDouble: [ref.types.ulong, [
    THFilePtr,
    THDoubleStoragePtr,
  ]],
  THFile_readByteRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.uchar),
    ref.types.ulong,
  ]],
  THFile_readCharRaw: [ref.types.ulong, [
    THFilePtr,
    ref.types.CString,
    ref.types.ulong,
  ]],
  THFile_readShortRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.short),
    ref.types.ulong,
  ]],
  THFile_readIntRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.int32),
    ref.types.ulong,
  ]],
  THFile_readLongRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.long),
    ref.types.ulong,
  ]],
  THFile_readFloatRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.float),
    ref.types.ulong,
  ]],
  THFile_readDoubleRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.double),
    ref.types.ulong,
  ]],
  THFile_readStringRaw: [ref.types.ulong, [
    THFilePtr,
    ref.types.CString,
    voidPtr,
  ]],
  THFile_writeByteRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.uchar),
    ref.types.ulong,
  ]],
  THFile_writeCharRaw: [ref.types.ulong, [
    THFilePtr,
    ref.types.CString,
    ref.types.ulong,
  ]],
  THFile_writeShortRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.short),
    ref.types.ulong,
  ]],
  THFile_writeIntRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.int32),
    ref.types.ulong,
  ]],
  THFile_writeLongRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.long),
    ref.types.ulong,
  ]],
  THFile_writeFloatRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.float),
    ref.types.ulong,
  ]],
  THFile_writeDoubleRaw: [ref.types.ulong, [
    THFilePtr,
    ref.refType(ref.types.double),
    ref.types.ulong,
  ]],
  THFile_writeStringRaw: [ref.types.ulong, [
    THFilePtr,
    ref.types.CString,
    ref.types.ulong,
  ]],
  THFile_readHalfScalar: [THHalf, [
    THFilePtr,
  ]],
  THFile_writeHalfScalar: [ref.types.void, [
    THFilePtr,
    THHalf,
  ]],
  THFile_readHalf: [ref.types.ulong, [
    THFilePtr,
    THHalfStoragePtr,
  ]],
  THFile_writeHalf: [ref.types.ulong, [
    THFilePtr,
    THHalfStoragePtr,
  ]],
  THFile_readHalfRaw: [ref.types.ulong, [
    THFilePtr,
    THHalfPtr,
    ref.types.ulong,
  ]],
  THFile_writeHalfRaw: [ref.types.ulong, [
    THFilePtr,
    THHalfPtr,
    ref.types.ulong,
  ]],
  THFile_synchronize: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_seek: [ref.types.void, [
    THFilePtr,
    ref.types.ulong,
  ]],
  THFile_seekEnd: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_position: [ref.types.ulong, [
    THFilePtr,
  ]],
  THFile_close: [ref.types.void, [
    THFilePtr,
  ]],
  THFile_free: [ref.types.void, [
    THFilePtr,
  ]],
  THDiskFile_new: [THFilePtr, [
    ref.types.CString,
    ref.types.CString,
    ref.types.int32,
  ]],
  THPipeFile_new: [THFilePtr, [
    ref.types.CString,
    ref.types.CString,
    ref.types.int32,
  ]],
  THDiskFile_name: [ref.types.CString, [
    THFilePtr,
  ]],
  THDiskFile_isLittleEndianCPU: [ref.types.int32, [
  ]],
  THDiskFile_isBigEndianCPU: [ref.types.int32, [
  ]],
  THDiskFile_nativeEndianEncoding: [ref.types.void, [
    THFilePtr,
  ]],
  THDiskFile_littleEndianEncoding: [ref.types.void, [
    THFilePtr,
  ]],
  THDiskFile_bigEndianEncoding: [ref.types.void, [
    THFilePtr,
  ]],
  THDiskFile_longSize: [ref.types.void, [
    THFilePtr,
    ref.types.int32,
  ]],
  THDiskFile_noBuffer: [ref.types.void, [
    THFilePtr,
  ]],
  THMemoryFile_newWithStorage: [THFilePtr, [
    THCharStoragePtr,
    ref.types.CString,
  ]],
  THMemoryFile_new: [THFilePtr, [
    ref.types.CString,
  ]],
  THMemoryFile_storage: [THCharStoragePtr, [
    THFilePtr,
  ]],
  THMemoryFile_longSize: [ref.types.void, [
    THFilePtr,
    ref.types.int32,
  ]],
});


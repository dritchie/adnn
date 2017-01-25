var Tensor = require("./tensor.js");
var dx = require("./ad/derivatives.js")


var ATT = require('adnn/tensor');
var CUTT = Tensor

var s1 = 10
var tt = new Tensor([s1,s1+2])
var ttclone = new Tensor([s1,s1+2])
// var tt = new Tensor([s1,s1,s1])

// console.log("tt.data ", tt)

// for(var i=0; i < 10; i++)
// {
//    console.log("Fill & sum: ", tt.fill(i).sum())
// }

// // var attempt_resize =
// tt.reshape([2,5,10])
// console.log("viewed: ", tt)

// for(var i=0; i < 10; i++)
// {
//    console.log("Fill & sum: ", tt.fill(i).sum())
// }

// fill it with small things again
tt.fill(.1)

console.log("Filled & summed: ", tt.sum())

// double it
tt.apply_function(function(val)
{
   // console.log('call: ', val)
   return val*2
})

console.log("Filled & summed: ", tt.sum())

for(var i=0; i < 5; i++){
   tt.fillRandom()
   console.log("Filled & summed random: ", tt.sum())
   ttclone.copy(tt)
   console.log("copied sum: ", ttclone.sum())
   console.log("Cloned sum: ", tt.clone().sum())
}
// console.log(tt.toString())
console.log("TTS0:", tt.size(0))
console.log("TTS1:", tt.size(1))
console.log("TTSA:", tt.size())

tt.fill(.05)


console.log("Min first: ", tt.min())
console.log("Max first: ", tt.max())

console.log("get 1:",tt.get([1,1]))
console.log("set return: ", tt.set([1,1], .123))
console.log("get2:", tt.get([1,1]))

console.log("Min now: ", tt.min())
console.log("Max now: ", tt.max())

console.log("all/any")
console.log("All nonzero: ", tt.all())
tt.set([1,1], 0)
console.log("All nonzero after 1 zero: ", tt.all())
tt.fill(0)
console.log("Any nonzero after all zero: ", tt.any())
tt.set([1,1], 1)
console.log("Any nonzero after 1 set: ", tt.any())


//create some new tensors, multiple together
var aa = new Tensor([2,4]).fill(.1)
var bb = new Tensor([4,2]).fill(2)
var cc = aa.dot(bb)


console.log("Check mm (~4.8): ", cc.sum())


var rr = new Tensor([5,5]).zero().addeq(.1).subeq(.3).muleq(0).addeq(2).poweq(2).subeq(.01).diveq(1.5)
console.log("Add/sub in place: ", rr.sum(), " math should be", (5*5*(4-.01)/1.5))


console.log("-1*2*4", aa.fill(1).negeq().sum())
console.log("1*4*3", bb.fill(.9).roundeq().sum())
console.log("Math.log(10)*4*3", bb.fill(10).logeq().sum(), " act: ", (Math.log(10)*4*3))
console.log("e*4*3", bb.fill(1).expeq().sum(), " act: ", (Math.exp(1)*4*3))
console.log("sqrt(16)*4*3", bb.fill(16).sqrteq().sum(), " act: ", (Math.sqrt(16)*4*3))
console.log("abs(-2.2)*4*3", bb.fill(-2.2).abseq().sum(), " act: ", (Math.abs(-2.2)*4*3))
console.log("ceil(2.2)*4*3", bb.fill(2.2).ceileq().sum(), " act: ", (Math.ceil(2.2)*4*3))
console.log("floor(1.2)*4*3", bb.fill(1.2).flooreq().sum(), " act: ", (Math.floor(1.2)*4*3))
console.log("cos(2)*4*3", bb.fill(2).coseq().sum(), " act: ", (Math.cos(2)*4*3))

var ccATT = new ATT(cc.dims);

console.log("Determining determinant")
console.log("Rand Determinant: ", cc.fillRandom().determinant())

cc.fill(1.25)
ccATT.fill(1.25)

var ii = 22
for(var i=0; i < cc.dims[0]; i++)
{
   for(var j=0; j < cc.dims[1]; j++)
   {
      cc.set([i,j], ii);
      // console.log("i,j", i,j, " val:", cc.get([i,j]))
      ccATT.data[i*cc.dims[1] + j] = ii;
      ii+= 12.1;
   }
}

// for(var ca =0; ca < ccATT.length; ca++)
//    console.log("ccATT ", ca, " val: ", ccATT.data[ca])
console.log("Filled Determinant: ", cc.determinant())
console.log("Filled Determinant adnn: ", ccATT.determinant())

// addUnaryMethod('neg', '-x');
// addUnaryMethod('round', 'Math.round(x)');
// addUnaryMethod('log', 'Math.log(x)');
// addUnaryMethod('exp', 'Math.exp(x)');
// addUnaryMethod('sqrt', 'Math.sqrt(x)');
// addUnaryMethod('abs', 'Math.abs(x)');
// addUnaryMethod('ceil', 'Math.ceil(x)');
// addUnaryMethod('floor', 'Math.floor(x)');
// addUnaryMethod('cos', 'Math.cos(x)');
// addUnaryMethod('sin', 'Math.sin(x)');
// addUnaryMethod('tan', 'Math.tan(x)');
// addUnaryMethod('acos', 'Math.acos(x)');
// addUnaryMethod('asin', 'Math.asin(x)');
// addUnaryMethod('atan', 'Math.atan(x)');
// addUnaryMethod('cosh', 'Math.cosh(x)');
// addUnaryMethod('sinh', 'Math.sinh(x)');
// addUnaryMethod('tanh', 'Math.tanh(x)');
// addUnaryMethod('acosh', 'Math.acosh(x)');
// addUnaryMethod('asinh', 'Math.asinh(x)');
// addUnaryMethod('atanh', 'Math.atanh(x)');
// addUnaryMethod('sigmoid', '1 / (1 + Math.exp(-x))');
// addUnaryMethod('isFinite', 'isFinite(x)');
// addUnaryMethod('isNaN', 'isNaN(x)');
// addUnaryMethod('invert', '1/x');
// addUnaryMethod('pseudoinvert', 'x === 0 ? 0 : 1/x');



// addReduction('sum', '0', 'accum + x');
// addReduction('min', 'Infinity', 'Math.min(accum, x)');
// addReduction('max', '-Infinity', 'Math.max(accum, x)');
// addReduction('all', 'true', 'accum && (x !== 0)');
// addReduction('any', 'false', 'accum || (x !== 0)');


// var dd = new Tensor([5,5]).fillRandom().addeq(1)
// console.log("Add min: " , dd.min(), " max ;" , dd.max())
// var ddchol = dd.cholesky()

// console.log("Choho: ", ddchol.sum())



// console.log("Filling data")
// tt.fill(.15)

// console.log("Summing data")
// console.log("summed: " ,  tt.sum())


// return

// var MTH = require("./torch.js")

// // console.log("Initialized? ", cuda_state)
// // module.exports = MTH


// var cute = MTH.THCudaTensor_newWithSize()
// var rs = MTH.THCudaTensor_resize(cute, MTH.THLongStorage_newWithSize)



// // Benchmarking ffi overheads
// var SZ1 = 128*3
// var SZ2 = 64//= 10
// var SZ3 = 64//= 10
// var N = 1000

// function creatv(SZ1, SZ2, SZ3, isvec, use_cuda) {
//    if (isvec) {
//    	SZ1 = SZ1 * SZ2 * SZ3
//    	SZ2 = undefined
//    	SZ3 = undefined
//    }
//    var t
//    if (SZ2 && SZ3) {
// 	  t = use_cuda ? MTH.THCudaTensor_newWithSize3d(SZ1, SZ2, SZ3) :
//                      MTH.THFloatTensor_newWithSize3d(SZ1, SZ2, SZ3)

//    } else if (SZ2) {
//    	t = use_cuda ? MTH.THCudaTensor_newWithSize2d(SZ1, SZ2):
//                      MTH.THFloatTensor_newWithSize2d(SZ1, SZ2)
//    	}
//    else {
//    	t = use_cuda ? MTH.THCudaTensor_newWithSize1d(SZ1):
//                      MTH.THFloatTensor_newWithSize1d(SZ1)
//    	}

//    if(use_cuda) MTH.THCudaTensor_fill( t, 0.15)
//    else
//          MTH.THFloatTensor_fill(t, 0.15)

//    return t
// }


// function creatett(SZ1, SZ2, SZ3, isvec) {

//    if (isvec) {
//       SZ1 = SZ1 * SZ2 * SZ3
//       SZ2 = undefined
//       SZ3 = undefined
//    }

//    var t
//    if (SZ2 && SZ3) {
//    t = new ATT([SZ1, SZ2, SZ3])
//    } else if (SZ2) {
//       t = new ATT([SZ1, SZ2])
//       }
//    else {
//       t = new ATT([SZ1])
//       }

//    t.fill(.15)
//    // MTH.THFloatTensor_fill(t, 0.15)
//    return t
// }


// var x = creatv(SZ1, SZ2, SZ3)
// var y = creatv(SZ1, SZ2, SZ3, true)
// var z = creatv(SZ1, SZ2, SZ3, true)

// var cux = creatv(SZ1, SZ2, SZ3, undefined, true)
// var cuy = creatv(SZ1, SZ2, SZ3, true, true)
// var cuz = creatv(SZ1, SZ2, SZ3, true, true)

// var ttx = creatett(SZ1, SZ2, SZ3)
// var tty = creatett(SZ1, SZ2, SZ3, true)
// var ttz = creatett(SZ1, SZ2, SZ3, true)


var b1 = 1
var s1 = 4
var s2 = 8
var s3 = 8

var ppltt = new ATT([b1,s1,s2,s3])
var cutt = new CUTT([b1,s1,s2,s3])
N = 1000;

function benchmark(txt, func, endfunc, tt, tt2)   {
   console.log('--------------------------------------------')
   console.log(txt + ' input size: ' + tt.length + ' iterations: ' + N + ' ')
   console.time(txt)
   for(i=1; i<N; i++) {
      func(tt, tt2)
  }
   console.timeEnd(txt)
   endfunc()
}

// benchmark(
//    'torch7 sumall',
//    function(tt) {
//       // sum = MTH.THFloatTensor_sumall(x)



//    },
//    function() {
//       console.log('finished cutorch')
//    },
//    cutt
// )

function tensor_test(tt, tt2) {
   // sum = MTH.THCudaTensor_sumall( cux)
   // tt.fill(Math.random()).sumreduce()
   // tt.fill(Math.random()).softmax()
   tt.fill(1).negeq().roundeq().logeq().expeq().abseq()
   tt.fill(1.5).sumreduce()
   tt.fill(2.1)
   .fill(Math.random()).coseq()
   .fill(Math.random()).sineq()
   .fill(Math.random()).cosheq()
   .fill(Math.random()).sinheq()
   .fill(Math.random()).abseq().addeq(.1).sqrteq()

   tt.fill(Math.random())
   .addeq(tt2.fill(Math.random()))
   .subeq(tt2)
   .muleq(tt2.fill(3))
   .muleq(1.5)
   .diveq(tt2.fill(1.5))
   .diveq(.5)
   .poweq(tt2.fill(2))
   .poweq(2)

   tt.fill(1).negeq()
   .fill(1).roundeq()
   .fill(10).logeq()
   .fill(1).expeq()
   .fill(-1).abseq()
   .fill(2.1).ceileq()
   .fill(4.2).flooreq()
   .fill(2).coseq()
   .fill(1).roundeq()
   .fill(10).logeq()
   .fill(1).expeq()
   .fill(-1).abseq()
   .fill(2.1).ceileq()
   .fill(4.2).flooreq()
   .fill(2).coseq()
}


// console.log()
benchmark(
   'cutorch7 math',
   tensor_test,
   function() {
      // console.log('sum: ' + sum)
      console.log('finished cutorch')
   },
   cutt, cutt.clone()
)
benchmark(
   'webppl math',
   tensor_test,
   function() {
      console.log('finished webppl')
      // console.log('sum: ' + sum)
   },
   ppltt, ppltt.clone()
)




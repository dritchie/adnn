var ffith=require('/Users/jpchen/jstorch/torch.js/TH.js')
var TH = ffith.TH

console.log('creating new 10-element tensor')
var tensor1=TH.THFloatTensor_newWithSize1d(10)

console.log('filling tensor with 0.15')
TH.THFloatTensor_fill(tensor1, 0.15)

console.log('printing tensor')
for (i=0; i < 10; ++i) {
    console.log(TH.THFloatTensor_get1d(tensor1, i))
}

// Benchmarking ffi overheads
var SZ1 = 10
var SZ2 //= 10
var SZ3 //= 10
var N = 10000000

function creatv(SZ1, SZ2, SZ3, isvec) {
   if (isvec) {
   	SZ1 = SZ1 * SZ2 * SZ3
      console.log("ok", SZ1);
   	SZ2 = undefined
   	SZ3 = undefined
   }   
   var t
   if (SZ2 && SZ3) {
	t = TH.THFloatTensor_newWithSize3d(SZ1, SZ2, SZ3)
   } else if (SZ2) {
   	t = TH.THFloatTensor_newWithSize2d(SZ1, SZ2)      
   	}
   else {
   	t = TH.THFloatTensor_newWithSize1d(SZ1)
   	}   
   // console.log(t)
   for (i=0; i < 10; ++i) {
    console.log("1st", TH.THFloatTensor_get1d(t, i))
   }
   console.log(t);
   TH.THFloatTensor_fill(t, 0.15)
   for (i=0; i < 10; ++i) {
    console.log("2nd", TH.THFloatTensor_get1d(t, i))
   }
   console.log(t);
   return t
}

var x = creatv(SZ1, SZ2, SZ3)
return;
var y = creatv(SZ1, SZ2, SZ3, true)
var z = creatv(SZ1, SZ2, SZ3, true)

function benchmark(txt, func, endfunc)   {
   console.log('--------------------------------------------')   
   console.log(txt + ' input size: ' + SZ1 + ' iterations: ' + N + ' ')
   console.time(txt)
   for(i=1; i<N; i++) {
      func()
  }
   console.timeEnd(txt)
   endfunc()
}

benchmark(
   'torch7 sumall',   
   function() {
      sum = TH.THFloatTensor_sumall(x)
   },
   function() {
      console.log('sum: ' + sum)
   }
)

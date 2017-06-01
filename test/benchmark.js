var Tensor = require('../THTensor');
var oldTensor = require('../tensor');
// var ad = require('../ad/adjs');
// var nn = require('../nn');

var t_0, t_1
var t_0 = new Tensor([3, 3]).fromArray([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
console.log(t_0.toArray())
console.log(t_0.cholesky().toArray()); return
var z = new Tensor([1000]);
a = []
var z_0 = new Tensor([4]);
var y = z_0.fromArray([1,2,3,4]);
// console.log(y.diagonal()); return
for (var i =0; i < 1000; i++){
  a.push(i)
}
var zz = z.fromArray(a);
console.log(zz.get([990])); 
console.log(zz.concat(zz).toArray()); return
var z = new Tensor([2,2]);
var z_0 = new tensor([3,2]);
var x = z_0.fromarray([[1,2],[3,4], [5,6]]);
var y = x.reshape([2,3])
console.log(y.toArray())
// return;
// var x_0 = z_0.fromArray([[3,5],[1,1]]);
// var y = x.ge(x_0)
// console.log(y.printByteArray())

function init(N){
  var n = N;
  t_0 = new Tensor([N,N]).fillRandom();
  t_1 = new Tensor([N,N]).fillRandom();
  return t_0, t_1
}

function matrixOps(m1, m2){
  m1.sumreduce();
  m1.inverse();
  m1.determinant();
  m1.cholesky();
  m1.transpose();
  m1.diagonal();
  m1.dot(m2);
  console.log("matrix Ops complete")
  return 0;
}

function mathOps(m1,m2){
//   m1.sum(m2);
  m1.sub(m2);
  m1.mul(m2);
  m1.div(m2);
  m1.ge(m2);
  m1.gt(m2);
  m1.eq(m2);
  m1.le(m2);
  m1.lt(m2);
  m1.sin();
  m1.sinh();
  m1.tanh();
  m1.atanh();
  m1.abs();
  m1.floor();
  m1.ceil();
  m1.log();
  m1.exp();
  m1.sqrt();
  m1.round();
  console.log("mathOps complete")
  return 0;
}
// var t_1 = new Tensor([3,3]);
// var dat = t_1.fromArray([[4.5,1.4,2.1],[-3,2,1],[-2,0.3,9.2]])
// console.log(x.determinant());
var N = [50, 100, 200, 400, 600, 1000]
for (var i = 0; i < N.length; i++) {
var mat_1 = new oldTensor([128,N[i]]).fillRandom();
var mat_2 = new oldTensor([N[i],N[i]]).fillRandom();
// mathOps(mat_1, mat_2);
// matrixOps(x, mat_2);
//console.log(mat.toArray());
//var ind = new Tensor([N,N]).fill(2.8);
//ind.fromArray([[0,1],[2,3]]);
//ind.set([0,0], 0.666);
//console.log("ind", ind.toArray());
//console.log("filled", mat.fill(3.0).data);
//console.log(mat.ls_to_array(mat.size));
// console.log(mat.toFlatArray());
// console.log(mat.sum());
//var mat_1 = new Tensor([N,N]).fillRandom();
// var mat_1 = new oldTensor([N,N]).fill(2.2);
console.log('starting dot')
console.time('tensor');
//var res = data.min();
//
//
//var res = mat.mul(mat_1);
var res = mat_1.dot(mat_2);

// Train
//var trainFunc = nn.linear(5, 5);
//console.time('tensor');
//opt.nnTrain(trainFunc, data, opt.regressionLoss, {
//	iterations: 10000,
//	batchSize: 1,
//	method: opt.sgd({ stepSize: 1, stepSizeDecay: 0.999 }),
//	// method: opt.adagrad({ stepSize: 1 }),
//	// method: opt.rmsprop(),
//	// method: opt.adam(),
//	verbose: false
//});
console.timeEnd('tensor');
}
// console.log(res.toFlatArray());
return

function benchmark(txt, func, endfunc)   {
   console.log('--------------------------------------------')   
   console.log(txt + ' input size: ' + SZ1 + ' iterations: ' + N + ' ')
   console.time(txt)
   for(i=1; i<N; i++) {
      func();
  }
   console.timeEnd(txt)
   endfunc();
}

benchmark(
   'torch7 tensorOps',   
   function() {
      tensorOps(m1,m2);
   }
)

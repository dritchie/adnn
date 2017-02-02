var Tensor = require('../newtensor');
//var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');
var opt = require('../opt');
var profiler = require('gc-profiler');

//Profile GC
// node --inspect test/gc.js

var numT = 10000;
var sizeT = 1000
console.time('tensor');
profiler.on('gc', function (info) {
  console.log(info);
});
while (numT-- > 0) {
  var t = new Tensor([sizeT,sizeT]).fill(0);
}
console.timeEnd('tensor');

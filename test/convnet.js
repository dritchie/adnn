var Tensor = require('../THTensor');
//var Tensor = require('../tensor');
var ad = require('../newad');
var nn = require('../nn');
var opt = require('../opt');

console.time('cnn');
var net = nn.sequence([
  // Assumes inputs are 32x32 RGB images (i.e. 3x32x32 Tensors)
  nn.convolution({inDepth: 3, outDepth: 16, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 16x16x16
  nn.convolution({inDepth: 16, outDepth: 20, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 20x8x8
  nn.convolution({inDepth: 20, outDepth: 20, filterSize: 5}),
  nn.relu,
  nn.maxpool({filterSize: 2}),
  // Data now has size 20x4x4 = 320
  nn.linear(320, 10),
  nn.softmax
  // Output is 10 class probabilities
]);
console.timeEnd('cnn');

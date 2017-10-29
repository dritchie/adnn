var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');
var opt = require('../opt');

var nInputs = 6
var nHidden = [4, 4, 5]
var nOutput = 2

var net = nn.sequence([
  nn.linear(nInputs, nHidden[0]),
  nn.sigmoid,
  nn.linear(nHidden[0], nHidden[1]),
  nn.sigmoid,
  nn.linear(nHidden[1], nOutput),
  nn.softmax
])

var data = [
  {input: [0.4, 0.5, 0.5, 0.,  0.,  0.], output: [1]},
  {input: [0.5, 0.3, 0.5, 0.,  0.,  0.], output: [1]},
  {input: [0.4, 0.5, 0.5, 0.,  0.,  0.], output: [1]},
  {input: [0.,  0.,  0.5, 0.3, 0.5, 0.], output: [0]},
  {input: [0.,  0.,  0.5, 0.4, 0.5, 0.], output: [0]},
  {input: [0.,  0.,  0.5, 0.5, 0.5, 0.], output: [0]}
]

var trainingData = nn.loadData(data)

console.log('nnTrain')
opt.nnTrain(net, trainingData, opt.classificationLoss, {
  batchSize: 1, // batch 超過 3 就無法成功， why ?
  iterations: 1000,
  method: opt.sgd({ stepSize: 1, stepSizeDecay: 0.999 }),
  verbose: true
})

console.log('predict')
// Predict class probabilities for new, unseen features
for (let i = 0; i < trainingData.length; i++) {
  var input = trainingData[i].input
  var probs = net.eval(input)
  console.log('input=%j\noutput=%j', input, probs)
}

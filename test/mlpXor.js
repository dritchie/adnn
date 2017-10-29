var Tensor = require('../tensor');
var ad = require('../ad');
var nn = require('../nn');
var opt = require('../opt');

var nInputs = 2
var nHidden = 5
var nClasses = 2

var net = nn.sequence([
  nn.linear(nInputs, nHidden),
  nn.tanh,
  nn.linear(nHidden, nClasses),
  nn.softmax
])

var data = [{ input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }]

var trainingData = nn.loadData(data)

console.log('nnTrain')
opt.nnTrain(net, trainingData, opt.classificationLoss, {
  batchSize: 2,
  iterations: 500,
  method: opt.sgd({ stepSize: 1, stepSizeDecay: 0.999 }),
  verbose: true
})

console.log('predict')
// Predict class probabilities for new, unseen features
for (let i = 0; i < trainingData.length; i++) {
  var input = trainingData[i].input
  var probs = net.eval(input)
  console.log('input=%j\nprobs=%j', input, probs)
}

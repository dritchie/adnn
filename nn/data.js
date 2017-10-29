var Tensor = require('../tensor.js');
var U = module.exports = {}

U.toTensor = function (dim, flatArray) {
  return new Tensor(dim).fromFlatArray(flatArray)
}

U.tensor1d = function (array) {
  return new Tensor([array.length]).fromFlatArray(array)
}

U.loadData = function (data) {
  var tensorData = []
  for (var i = 0; i < data.length; i++) {
    var x = U.tensor1d(data[i].input)
    var y = U.tensor1d(data[i].output)
    tensorData.push({
      input: x,
      output: y
    })
  }
  return tensorData
}

var graph = require('./graph.js');
var func = require('./func.js');
var functions = require('./functions.js');
var Tensor = require('../tensor.js');

function liftScalar(x) { return new graph.ScalarNode(x); };
function liftTensor(x) { return new graph.TensorNode(x); };
function doLift(x) {
	return x instanceof Tensor ? liftTensor(x) : liftScalar(x);
}

var ad = {
	lift: function(x) { return graph.isNode(x) ? x : doLift(x); },
	isLifted: graph.isNode,
	project: function(x) { return graph.isNode(x) ? x.x : x; }
};
for (var prop in graph) {
	ad[prop] = func[prop];
	ad[prop] = functions[prop];
}

module.exports = ad;
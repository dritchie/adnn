var graph = require('./graph.js');
var func = require('./func.js');
var functions = require('./functions.js');

var ad = {
	liftScalar: function(x) { return new graph.ScalarNode(x); },
	liftTensor: function(x) { return new graph.TensorNode(x); },
	isLifted: graph.isNode,
	project: function(x) { return graph.isNode(x) ? x.x : x; }
};
for (var prop in graph) {
	ad[prop] = func[prop];
	ad[prop] = functions[prop];
}

module.exports = ad;
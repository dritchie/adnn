var graph = require('./graph.js');
var func = require('./func.js');
var functions = require('./functions.js');
var Tensor = require('../tensor.js');

var ad = {
	lift: function(x) {
		return x instanceof Tensor ?
			new graph.TensorNode(x) :
			new graph.ScalarNode(x);
	},
	isLifted: graph.isNode,
	project: function(x) { return x.x; }
};
for (var prop in graph) {
	ad[prop] = func[prop];
	ad[prop] = functions[prop];
}

module.exports = ad;
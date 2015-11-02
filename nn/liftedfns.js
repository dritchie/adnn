var adfunctions = require('../ad/functions.js');
var lift = require('./lift.js');

module.exports = {};

// We go ahead and lift all of the Tensor-valued AD functions in
//    ad/functions.js
for (var fnname in adfunctions.tensor) {
	module.exports[fnname] = lift(adfunctions.tensor[fnname]);
}
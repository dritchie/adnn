var adfunctions = require('../../ad/functions.js');
var lift = require('../lift.js');

// We go ahead and lift all of the Tensor-valued AD functions in
//    ad/functions.js
module.exports = {};
for (var fnname in adfunctions.tensor) {
	module.exports[fnname] = lift(adfunctions.tensor[fnname], fnname);
}
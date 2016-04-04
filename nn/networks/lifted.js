'use strict';

var adfunctions = require('../../ad/functions.js');
var lifting = require('../lifting.js');


// We go ahead and lift all of the Tensor-valued AD functions in
//    ad/functions.js
module.exports = {};
for (var fnname in adfunctions.tensor) {
	module.exports[fnname] = lifting.lift(adfunctions.tensor[fnname], fnname);
}


// A few of these functions make more sense when partially-evaluated
module.exports.range = lifting.partialeval(adfunctions.tensor.range, 'range');
module.exports.split = lifting.partialeval(adfunctions.tensor.split, 'split');
module.exports.reshape = lifting.partialeval(adfunctions.tensor.reshape, 'reshape');

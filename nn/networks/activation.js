var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var lift = require('../lift.js');


// The 'lifted' module already defines sigmoid and tanh.


var relu = lift(ad.newUnaryFunction({
	OutputType: Tensor,
	name: 'relu',
	forward: function(x) {
		x = ad.project(x);
		var y = x.clone();
		var n = x.length;
		while (n--) {
			y.data[n] = y.data[n] < 0 ? 0 : y.data[n];
		}
		return y;
	},
	backward: function(x) {
		var n = x.x.length;
		while (n--) {
			x.dx.data[n] += x.x.data[n] <= 0 ? 0 : this.dx.data[n];
		}
	}
}), 'relu');


module.exports = {
	relu: relu
};
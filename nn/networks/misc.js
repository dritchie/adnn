'use strict';

var ad = require('../../ad');
var Network = require('../network.js');


// Network that takes no inputs and returns a constant set of parameters
function ConstantParamNetwork(dims, optname) {
	Network.call(this);
	this.name = optname || 'constantparams';
	this.dims = dims;

	this.params = ad.params(dims, this.name);
	this.paramGetters = [
		function() { return this.params; }.bind(this)
	];
	this.paramSetters = [
		function(params) { this.params = params; }.bind(this)
	];
};
ConstantParamNetwork.prototype = Object.create(Network.prototype);
ConstantParamNetwork.prototype.eval = function() {
	return this.isTraining ? this.params : ad.value(this.params);
};
ConstantParamNetwork.prototype.serializeJSON = function() {
	return {
		type: 'constantparams',
		name: this.name,
		dims: this.dims,
		params: ad.value(this.params).toFlatArray()
	};
};
Network.deserializers.constantparams = function(json) {
	var net = new ConstantParamNetwork(json.dims, json.name);
	ad.value(net.params).fromFlatArray(json.params);
	return net;
};
function constantparams(dims, optname) {
	return new ConstantParamNetwork(dims, optname);
}


module.exports = {
	constantparams: constantparams
};

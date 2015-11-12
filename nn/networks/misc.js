var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var Network = require('../network.js');


// Network that takes no inputs and returns a constant set of parameters
function ConstantParamNetwork(dims, optname) {
	Network.call(this);
	this.name = optname || 'constantparams';
	this.parameters = [ad.lift(new Tensor(dims).fillRandom(), this.name)];
	this.isTraining = false;
};
ConstantParamNetwork.prototype = Object.create(Network.prototype);
ConstantParamNetwork.prototype.eval = function() {
	return this.isTraining ? this.parameters[0] : ad.project(this.parameters[0]);
};
ConstantParamNetwork.prototype.setTraining = function(flag) {
	this.isTraining = flag;
};
ConstantParamNetwork.prototype.serializeJSON = function() {
	return {
		type: 'constantparams',
		name: this.name,
		dims: this.dims,
		params: ad.project(this.parameters[0]).toFlatArray()
	};
};
Network.deserializers.constantparams = function(json) {
	var net = new ConstantParamNetwork(json.dims, json.name);
	ad.project(net.parameters[0]).fromFlatArray(json.params);
	return net;
};
function constantparams(dims, optname) {
	return new ConstantParamNetwork(dims, optname);
}


module.exports = {
	constantparams: constantparams
};

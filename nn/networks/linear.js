'use strict';

var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var Network = require('../network.js');
var assert = require('assert');


// Fully connected network
function LinearNetwork(nIn, nOut, optname) {
	Network.call(this);
	this.name = optname || 'linear';
	this.inSize = nIn;
	this.outSize = nOut;

	this.weights = ad.params([nOut, nIn], this.name+'_weights');
	this.biases = ad.params([nOut], this.name+'_biases');
	this.paramGetters = [
		function() { return this.weights; }.bind(this),
		function() { return this.biases; }.bind(this)
	];
	this.paramSetters = [
		function(weights) { this.weights = weights; }.bind(this),
		function(biases) { this.biases = biases; }.bind(this)
	];
}
LinearNetwork.prototype = Object.create(Network.prototype);

LinearNetwork.prototype.serializeJSON = function() {
	return {
		type: 'linear',
		name: this.name,
		inSize: this.inSize,
		outSize: this.outSize,
		weights: ad.value(this.weights).toFlatArray(),
		biases: ad.value(this.biases).toFlatArray()
	};
}
Network.deserializers.linear = function(json) {
	var net = new LinearNetwork(json.inSize, json.outSize, json.name);
	ad.value(net.weights).fromFlatArray(json.weights);
	ad.value(net.biases).fromFlatArray(json.biases);
	return net;
};


var mvmuladd = ad.newFunction({
	OutputType: Tensor,
	name: 'mvmuladd',
	forward: function(A, x, b) {
		A = ad.value(A);
		x = ad.value(x);
		b = ad.value(b);
		var w = x.length;
		var h = b.length;
		if (w !== A.dims[1]) {
			assert(false, 'Linear network: input size is ' + w +
				' but should be ' + A.dims[1]);
		}
		var y = b.clone();
		for (var r = 0; r < h; r++) {
			var off = r*w;
			for (var c = 0; c < w; c++) {
				y.data[r] += A.data[off + c] * x.data[c];
			}
		}
		return y;
	},
	backward: function(A, x, b) {
		var Ap = ad.value(A);
		var xp = ad.value(x);
		var bp = ad.value(b);
		var aIs = A !== Ap;
		var xIs = x !== xp;
		var bIs = b !== bp;
		var w = xp.length;
		var h = bp.length;
		for (var r = 0; r < h; r++) {
			var off = r*w;
			var thisdx = this.dx.data[r];
			if (bIs) {
				b.dx.data[r] += thisdx;
			}
			for (var c = 0; c < w; c++) {
				if (xIs) {
					x.dx.data[c] += Ap.data[off + c] * thisdx;
				}
				if (aIs) {
					A.dx.data[off + c] += xp.data[c] * thisdx;
				}
			}
		}
	},
	getParents: ad.naryGetParents
});


LinearNetwork.prototype.eval = function(x) {
	var A = this.isTraining ? this.weights : ad.value(this.weights);
	var b = this.isTraining ? this.biases : ad.value(this.biases);
	return mvmuladd(A, x, b);
};


function linear(nIn, nOut, optname) {
	return new LinearNetwork(nIn, nOut, optname);
}


// ----------------------------------------------------------------------------


// Fully connected network between the layers (i.e. channels) of an image at 
//    each pixel
function LayerwiseLinearNetwork(nIn, nOut, optname) {
	Network.call(this);
	this.name = optname || 'layerwiseLinear';
	this.inSize = nIn;
	this.outSize = nOut;

	this.weights = ad.params([nOut, nIn], this.name+'_weights');
	this.biases = ad.params([nOut], this.name+'_biases');
	this.paramGetters = [
		function() { return this.weights; }.bind(this),
		function() { return this.biases; }.bind(this)
	];
	this.paramSetters = [
		function(weights) { this.weights = weights; }.bind(this),
		function(biases) { this.biases = biases; }.bind(this)
	];
}
LayerwiseLinearNetwork.prototype = Object.create(Network.prototype);

LayerwiseLinearNetwork.prototype.serializeJSON = function() {
	return {
		type: 'layerwiseLinear',
		name: this.name,
		inSize: this.inSize,
		outSize: this.outSize,
		weights: ad.value(this.weights).toFlatArray(),
		biases: ad.value(this.biases).toFlatArray()
	};
}
Network.deserializers.layerwiseLinear = function(json) {
	var net = new LayerwiseLinearNetwork(json.inSize, json.outSize, json.name);
	ad.value(net.weights).fromFlatArray(json.weights);
	ad.value(net.biases).fromFlatArray(json.biases);
	return net;
};


var layerwiseMVMulAdd = ad.newFunction({
	OutputType: Tensor,
	name: 'layerwiseMVMulAdd',
	forward: function(img, weights, biases) {
		img = ad.value(img);
		weights = ad.value(weights);
		biases = ad.value(biases);
		var inD = img.dims[0];
		if (inD !== weights.dims[1]) {
			assert(false, 'LayerwiseLinear Network: input depth is ' +
				inD + ' but should be ' + weights.dims[1]);
		}
		var h = img.dims[1];
		var w = img.dims[2];
		var outD = weights.dims[0];

		var outImg = new Tensor([outD, h, w]);

		for (var y = 0; y < h; y++) {
			for (var x = 0; x < w; x++) {
				for (var od = 0; od < outD; od++) {
					var outIdx = x+w*(y+h*od);
					outImg.data[outIdx] = biases.data[od];
					for (var id = 0; id < inD; id++) {
						var wIdx = id + od*inD;
						var wt = weights.data[wIdx];
						var inIdx = x+w*(y+h*id);
						var imval = img.data[inIdx];
						outImg.data[outIdx] += wt * imval;
					}
				}
			}
		}

		return outImg;
	},
	backward: function(img, weights, biases) {
		var imgp = ad.value(img);
		var weightsp = ad.value(weights);
		var biasesp = ad.value(biases);
		var imgIs = img !== imgp;
		var weightsIs = weights !== weightsp;
		var biasesIs = biases !== biasesp;
		var h = imgp.dims[1];
		var w = imgp.dims[2];
		var outD = weightsp.dims[0];
		var inD = weightsp.dims[1];

		for (var y = 0; y < h; y++) {
			for (var x = 0; x < w; x++) {
				for (var od = 0; od < outD; od++) {
					var outIdx = x+w*(y+h*od);
					var thisdx = this.dx.data[outIdx];
					if (biasesIs) {
						biases.dx.data[od] += thisdx;
					}
					for (var id = 0; id < inD; id++) {
						var wIdx = id + od*inD;
						var inIdx = x+w*(y+h*id);
						if (weightsIs) {
							var imval = imgp.data[inIdx];
							weights.dx.data[wIdx] += imval * thisdx;
						}
						if (imgIs) {
							var wt = weightsp.data[wIdx];
							img.dx.data[inIdx] += wt * thisdx;
						}
					}
				}
			}
		}
	},
	getParents: ad.naryGetParents
});


LayerwiseLinearNetwork.prototype.eval = function(img) {
	var weights = this.isTraining ? this.weights : ad.value(this.weights);
	var biases = this.isTraining ? this.biases : ad.value(this.biases);
	return layerwiseMVMulAdd(img, weights, biases);
};


function layerwiseLinear(nIn, nOut, optname) {
	return new LayerwiseLinearNetwork(nIn, nOut, optname);
}


// ----------------------------------------------------------------------------


module.exports = {
	linear: linear,
	layerwiseLinear: layerwiseLinear
};




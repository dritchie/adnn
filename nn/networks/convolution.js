'use strict';

var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var Network = require('../network.js');
var assert = require('assert');


// Adapted from:
// https://github.com/karpathy/convnetjs/blob/master/src/
//    convnet_layers_dotproducts.js


// Convolutions operate on rank-3 tensors.
// The dimensions of the tensor correspond to image dimensions as:
//    - dims[0]: depth/channel
//    - dims[1]: height
//    - dims[2]: width
// So essentially, images are stored as separate spatial slices for each
//    channel. This is not how convnetjs does it, but it appears to be
//    consistent with torch.

function ConvolutionNetwork(nIn, nOut, fW, fH, sX, sY, pX, pY, optname) {
	Network.call(this);
	this.name = optname || 'convolution';
	this.inDepth = nIn;
	this.outDepth = nOut;
	this.filterWidth = fW;
	this.filterHeight = fH;
	this.strideX = sX;
	this.strideY = sY;
	this.padX = pX;
	this.padY = pY;

	this.filters = ad.params([nOut, nIn, fH, fW], this.name+'_filters');
	this.biases = ad.params([nOut], this.name+'_biases');
	this.paramGetters = [
		function() { return this.filters; }.bind(this),
		function() { return this.biases; }.bind(this)
	];
	this.paramSetters = [
		function(filters) { this.filters = filters; }.bind(this),
		function(biases) { this.biases = biases; }.bind(this)
	];
}
ConvolutionNetwork.prototype = Object.create(Network.prototype);

ConvolutionNetwork.prototype.serializeJSON = function() {
	return {
		type: 'convolution',
		name: this.name,
		inDepth: this.inDepth,
		outDepth: this.outDepth,
		filterWidth: this.filterWidth,
		filterHeight: this.filterHeight,
		strideX: this.strideX,
		strideY: this.strideY,
		padX: this.padX,
		padY: this.padY,
		filters: ad.value(this.filters).toFlatArray(),
		biases: ad.value(this.biases).toFlatArray()
	};
}
Network.deserializers.convolution = function(json) {
	var net = new ConvolutionNetwork(json.inDepth, json.outDepth,
		json.filterWidth, json.filterHeight, json.strideX, json.strideY,
		json.padX, json.padY, json.name);
	ad.value(net.filters).fromFlatArray(json.filters);
	ad.value(net.biases).fromFlatArray(json.biases);
	return net;
};


var convolve = ad.newFunction({
	OutputType: Tensor,
	name: 'convolution',
	forward: function(inImg, filters, biases, strideX, strideY, padX, padY) {

		inImg = ad.value(inImg);
		filters = ad.value(filters);
		biases = ad.value(biases);


		var fH = filters.dims[2];
		var fW = filters.dims[3];
		var iD = inImg.dims[0];
		var iH = inImg.dims[1];
		var iW = inImg.dims[2];
		var oD = filters.dims[0];


		var oH = Math.floor((iH + 2*padY - fH) / strideY + 1);
		var oW = Math.floor((iW + 2*padX - fW) / strideX + 1);

		//console.log(oD, " ", oH, " ", oW);


		if (iD !== filters.dims[1]) {
			assert(false, 'Convolutional network: input depth is ' + iD +
				' but should be ' + filters.dims[1]); 
		}

		var outImg = new Tensor([oD, oH, oW]);

		for (var d = 0; d < oD; d++) {
			var x = -padX;
			var y = -padY;
			for (var ay = 0; ay < oH; y += strideY, ay++) {
				x = -padX;
				for (var ax = 0; ax < oW; x += strideX, ax++) {
					// Convolution
					var a = biases.data[d];
					for (var fd = 0; fd < iD; fd++) {
						for (var fy = 0; fy < fH; fy++) {
							var iy = y + fy;
							for (var fx = 0; fx < fW; fx++) {
								var ix = x + fx;
								if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
									var iIdx = ix+iW*(iy+iH*fd);
									var fIdx = fx+fW*(fy+fH*(fd+iD*d));
									a += filters.data[fIdx] * inImg.data[iIdx];
								}
							}
						}
					}
					// Set ouput pixel
					outImg.data[ax+oW*(ay+oH*d)] = a;
				}
			}
		}

		return outImg;
	},
	backward: function(inImg, filters, biases, strideX, strideY, padX, padY) {
		var inImgP = ad.value(inImg);
		var filtersP = ad.value(filters);
		var biasesP = ad.value(biases);
		var iIs = inImg !== inImgP;
		var fIs = filters !== filtersP;
		var bIs = biases !== biasesP;

		var fH = filtersP.dims[2];
		var fW = filtersP.dims[3];
		var iD = inImgP.dims[0];
		var iH = inImgP.dims[1];
		var iW = inImgP.dims[2];
		var oD = filtersP.dims[0];
		var oH = this.x.dims[1];
		var oW = this.x.dims[2];

		for (var d = 0; d < oD; d++) {
			var x = -padX;
			var y = -padY;
			for (var ay = 0; ay < oH; y += strideY, ay++) {
				x = -padX;
				for (var ax = 0; ax < oW; x += strideX, ax++) {
					var thisGrad = this.dx.data[ax+oW*(ay+oH*d)];
					// Convolution
					if (bIs) biases.dx.data[d] += thisGrad;
					for (var fd = 0; fd < iD; fd++) {
						for (var fy = 0; fy < fH; fy++) {
							var iy = y + fy;
							for (var fx = 0; fx < fW; fx++) {
								var ix = x + fx;
								if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
									var iIdx = ix+iW*(iy+iH*fd);
									var fIdx = fx+fW*(fy+fH*(fd+iD*d));
									if (iIs) {
										inImg.dx.data[iIdx] +=
											filtersP.data[fIdx]*thisGrad;
									}
									if (fIs) {
										filters.dx.data[fIdx] +=
											inImgP.data[iIdx]*thisGrad;
									}
								}
							}
						}
					}
				}
			}
		}
	},
	getParents: function(inImg, filters, biases, strideX, strideY, padX, padY) {
		var p = [];
		if (ad.isLifted(inImg)) p.push(inImg);
		if (ad.isLifted(filters)) p.push(filters);
		if (ad.isLifted(biases)) p.push(biases);
		return p;
	},
});


ConvolutionNetwork.prototype.eval = function(img) {
	var filters = this.isTraining ? this.filters : ad.value(this.filters);
	var biases = this.isTraining ? this.biases : ad.value(this.biases);
	return convolve(img, filters, biases,
		this.strideX, this.strideY, this.padX, this.padY);
};


function convolution(opts, optname) {
	var nIn = opts.inDepth || 1;
	var nOut = opts.outDepth || 1;
	var fW = opts.filterWidth || opts.filterSize;
	var fH = opts.filterHeight || opts.filterSize;
	var sX = opts.strideX || opts.stride || 1;
	var sY = opts.strideY || opts.stride || 1;
	var pX = opts.padX || opts.pad || 'same';
	var pY = opts.padY || opts.pad || 'same';
	// Can specify that output should be padded to be the same size as the input
	//    (assuming a stride of 1)
	if (pX === 'same') {
		pX = Math.floor((fW - 1)/2);
	}
	if (pY === 'same') {
		pY = Math.floor((fH - 1)/2);
	}
	return new ConvolutionNetwork(nIn, nOut, fW, fH, sX, sY, pX, pY, optname);
};


module.exports = {
	convolution: convolution, 
	convolve: convolve
};




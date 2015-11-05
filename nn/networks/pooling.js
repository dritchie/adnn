var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var Network = require('../network.js');

// Adapted from:
// https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_pool.js

// Images are represented as in convolution.js

function MaxPoolNetwork(fW, fH, sX, sY, pX, pY) {
	Network.call(this);
	this.filterWidth = fW;
	this.filterHeight = fH;
	this.strideX = sX;
	this.strideY = sY;
	this.padX = pX;
	this.padY = pY;
	this.name = 'maxpool';
}
MaxPoolNetwork.prototype = Object.create(Network.prototype);

MaxPoolNetwork.prototype.serializeJSON = function() {
	return {
		type: 'maxpool',
		filterWidth: this.filterWidth,
		filterHeight: this.filterHeight,
		strideX: this.strideX,
		strideY: this.strideY,
		padX: this.padX,
		padY: this.padY
	};
}
Network.deserializers.maxpool = function(json) {
	return new MaxPoolNetwork(json.filterWidth, json.filterHeight,
		json.strideX, json.strideY, json.padX, json.padY);
};


// Store the indices of where the maxes came from, for more efficient backprop
var maxIndices;
var maxpoolingImpl = ad.newFunction(Tensor, {
	forward: function(inImg, fW, fH, sX, sY, pX, pY) {
		inImg = ad.project(inImg);

		var D = inImg.dims[0];
		var iH = inImg.dims[1];
		var iW = inImg.dims[2];
		var oH = Math.floor((iH + 2*pY - fH) / sY + 1);
		var oW = Math.floor((iW + 2*pX - fW) / sX + 1);

		var outImg = new Tensor([D, oH, oW]);
		maxIndices = new Int32Array(D*oH*oW);

		for (var d = 0; d < D; d++) {
			var x = -padX;
			var y = -padY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -padX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Find max within filter window
					var maxval = -Infinity;
					var maxidx = -1;
					for (var fy = 0; fy < fH; fy++) {
						var iy = y + fy;
						for (var fx = 0; fx < fW; fx++) {
							var ix = x + fx;
							if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
								var idx = ix+iW*(iy+iH*fd);
								var v = inImg.data[idx];
								if (v > maxval) {
									maxval = v;
									maxidx = idx;
								}
							}
						}
					}
					// Set ouput pixel
					var outidx = ax+oW*(ay+oH*d);
					outImg.data[outidx] = maxval;
					maxIndices[outidx] = maxidx;
				}
			}
		}

		return outImg;
	},
	backward: function(inImg, fW, fH, sX, sY, pX, pY) {
		if (ad.isLifted(inImg)) {
			var D = inImg.x.dims[0];
			var iH = inImg.x.dims[1];
			var iW = inImg.x.dims[2];
			var oH = this.x.dims[1];
			var oW = this.x.dims[2];

			for (var d = 0; d < D; d++) {
				var x = -padX;
				var y = -padY;
				for (var ay = 0; ay < oH; y += sY, ay++) {
					x = -padX;
					for (var ax = 0; ax < oW; x += sX, ax++) {
						// Accumulate into correct derivative using maxIndices
						var outidx = ax+oW*(ay+oH*d);
						var inidx = this.maxIndices[outidx];
						inImg.dx.data[inidx] += this.dx.data[outidx];
					}
				}
			}
		}
	},
	getParents: function(inImg, fW, fH, sX, sY, pX, pY) {
		return ad.isLifted(inImg) ? [inImg] : [];
	}
});
function maxpooling(inImg, fW, fH, sX, sY, pX, pY) {
	var outImg = maxpoolingImpl(inImg, fW, fH, sX, sY, pX, pY);
	if (ad.isLifted(outImg)) {
		outImg.maxIndices = maxIndices;
	}
	return outImg;
}


MaxPoolNetwork.prototype.eval = function(img) {
	return maxpooling(img, this.filterWidth, this.filterHeight, this.strideX,
		this.strideY, this.padX, this.padY);
};


function maxpool(opts) {
	var fW = opts.filterWidth || opts.filterSize;
	var fH = opts.filterHeight || opts.filterSize;
	var sX = opts.strideX || opts.stride || 1;
	var sY = opts.strideY || opts.stride || 1;
	var pX = opts.padX || opts.pad || 0;
	var pY = opts.padY || opts.pad || 0;
	return new MaxPoolNetwork(fW, fH, sX, sY, pX, pY);
}


module.exports = {
	maxpool: maxpool
};




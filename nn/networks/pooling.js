'use strict';

var Tensor = require('../../tensor.js');
var ad = require('../../ad');
var Network = require('../network.js');


// Images are represented as in convolution.js


// Max pooling ----------------------------------------------------------------

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


// Adapted from:
// https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_pool.js
// Store the indices of where the maxes came from, for more efficient backprop
var maxIndices;
var maxpoolingImpl = ad.newFunction({
	OutputType: Tensor,
	name: 'maxpooling',
	forward: function(inImg, fW, fH, sX, sY, pX, pY) {
		inImg = ad.value(inImg);

		var D = inImg.dims[0];
		var iH = inImg.dims[1];
		var iW = inImg.dims[2];
		var oH = Math.floor((iH + 2*pY - fH) / sY + 1);
		var oW = Math.floor((iW + 2*pX - fW) / sX + 1);

		var outImg = new Tensor([D, oH, oW]);
		maxIndices = new Int32Array(D*oH*oW);

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Find max within filter window
					var maxval = -Infinity;
					var maxidx = -1;
					for (var fy = 0; fy < fH; fy++) {
						var iy = y + fy;
						for (var fx = 0; fx < fW; fx++) {
							var ix = x + fx;
							if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
								var idx = ix+iW*(iy+iH*d);
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
		// No need for an 'isLifted' check: there's only one possible lifted
		//    argument (inImg), so if we are backprop'ing, it must be lifted.
		var D = inImg.x.dims[0];
		var iH = inImg.x.dims[1];
		var iW = inImg.x.dims[2];
		var oH = this.x.dims[1];
		var oW = this.x.dims[2];

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Accumulate into correct derivative using maxIndices
					var outidx = ax+oW*(ay+oH*d);
					var inidx = this.maxIndices[outidx];
					inImg.dx.data[inidx] += this.dx.data[outidx];
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
	var sX = opts.strideX || opts.stride || fW;
	var sY = opts.strideY || opts.stride || fH;
	var pX = opts.padX || opts.pad || 0;
	var pY = opts.padY || opts.pad || 0;
	return new MaxPoolNetwork(fW, fH, sX, sY, pX, pY);
}


// Min pooling ----------------------------------------------------------------

function MinPoolNetwork(fW, fH, sX, sY, pX, pY) {
	Network.call(this);
	this.filterWidth = fW;
	this.filterHeight = fH;
	this.strideX = sX;
	this.strideY = sY;
	this.padX = pX;
	this.padY = pY;
	this.name = 'minpool';
}
MinPoolNetwork.prototype = Object.create(Network.prototype);

MinPoolNetwork.prototype.serializeJSON = function() {
	return {
		type: 'minpool',
		filterWidth: this.filterWidth,
		filterHeight: this.filterHeight,
		strideX: this.strideX,
		strideY: this.strideY,
		padX: this.padX,
		padY: this.padY
	};
}
Network.deserializers.minpool = function(json) {
	return new MinPoolNetwork(json.filterWidth, json.filterHeight,
		json.strideX, json.strideY, json.padX, json.padY);
};


var minIndices;
var minpoolingImpl = ad.newFunction({
	OutputType: Tensor,
	name: 'maxpooling',
	forward: function(inImg, fW, fH, sX, sY, pX, pY) {
		inImg = ad.value(inImg);

		var D = inImg.dims[0];
		var iH = inImg.dims[1];
		var iW = inImg.dims[2];
		var oH = Math.floor((iH + 2*pY - fH) / sY + 1);
		var oW = Math.floor((iW + 2*pX - fW) / sX + 1);

		var outImg = new Tensor([D, oH, oW]);
		minIndices = new Int32Array(D*oH*oW);

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Find min within filter window
					var minval = Infinity;
					var minidx = -1;
					for (var fy = 0; fy < fH; fy++) {
						var iy = y + fy;
						for (var fx = 0; fx < fW; fx++) {
							var ix = x + fx;
							if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
								var idx = ix+iW*(iy+iH*d);
								var v = inImg.data[idx];
								if (v < minval) {
									minval = v;
									minidx = idx;
								}
							}
						}
					}
					// Set ouput pixel
					var outidx = ax+oW*(ay+oH*d);
					outImg.data[outidx] = minval;
					minIndices[outidx] = minidx;
				}
			}
		}

		return outImg;
	},
	backward: function(inImg, fW, fH, sX, sY, pX, pY) {
		// No need for an 'isLifted' check: there's only one possible lifted
		//    argument (inImg), so if we are backprop'ing, it must be lifted.
		var D = inImg.x.dims[0];
		var iH = inImg.x.dims[1];
		var iW = inImg.x.dims[2];
		var oH = this.x.dims[1];
		var oW = this.x.dims[2];

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Accumulate into correct derivative using minIndices
					var outidx = ax+oW*(ay+oH*d);
					var inidx = this.minIndices[outidx];
					inImg.dx.data[inidx] += this.dx.data[outidx];
				}
			}
		}
	},
	getParents: function(inImg, fW, fH, sX, sY, pX, pY) {
		return ad.isLifted(inImg) ? [inImg] : [];
	}
});
function minpooling(inImg, fW, fH, sX, sY, pX, pY) {
	var outImg = minpoolingImpl(inImg, fW, fH, sX, sY, pX, pY);
	if (ad.isLifted(outImg)) {
		outImg.minIndices = minIndices;
	}
	return outImg;
}


MinPoolNetwork.prototype.eval = function(img) {
	return minpooling(img, this.filterWidth, this.filterHeight, this.strideX,
		this.strideY, this.padX, this.padY);
};


function minpool(opts) {
	var fW = opts.filterWidth || opts.filterSize;
	var fH = opts.filterHeight || opts.filterSize;
	var sX = opts.strideX || opts.stride || fW;
	var sY = opts.strideY || opts.stride || fH;
	var pX = opts.padX || opts.pad || 0;
	var pY = opts.padY || opts.pad || 0;
	return new MinPoolNetwork(fW, fH, sX, sY, pX, pY);
}


// Mean (i.e. average) pooling ------------------------------------------------


function MeanPoolNetwork(fW, fH, sX, sY, pX, pY) {
	Network.call(this);
	this.filterWidth = fW;
	this.filterHeight = fH;
	this.strideX = sX;
	this.strideY = sY;
	this.padX = pX;
	this.padY = pY;
	this.name = 'meanpool';
}
MeanPoolNetwork.prototype = Object.create(Network.prototype);

MeanPoolNetwork.prototype.serializeJSON = function() {
	return {
		type: 'meanpool',
		filterWidth: this.filterWidth,
		filterHeight: this.filterHeight,
		strideX: this.strideX,
		strideY: this.strideY,
		padX: this.padX,
		padY: this.padY
	};
}
Network.deserializers.meanpool = function(json) {
	return new MeanPoolNetwork(json.filterWidth, json.filterHeight,
		json.strideX, json.strideY, json.padX, json.padY);
};

var meanpooling = ad.newFunction({
	OutputType: Tensor,
	name: 'meanpooling',
	forward: function(inImg, fW, fH, sX, sY, pX, pY) {
		inImg = ad.value(inImg);

		var D = inImg.dims[0];
		var iH = inImg.dims[1];
		var iW = inImg.dims[2];
		var oH = Math.floor((iH + 2*pY - fH) / sY + 1);
		var oW = Math.floor((iW + 2*pX - fW) / sX + 1);
		var fN = fW*fH;

		var outImg = new Tensor([D, oH, oW]);

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					// Compute average within filter window
					var avgval = 0;
					for (var fy = 0; fy < fH; fy++) {
						var iy = y + fy;
						for (var fx = 0; fx < fW; fx++) {
							var ix = x + fx;
							if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
								var idx = ix+iW*(iy+iH*d);
								var v = inImg.data[idx];
								avgval += v;
							}
						}
					}
					// Set ouput pixel
					var outidx = ax+oW*(ay+oH*d);
					outImg.data[outidx] = avgval / fN;
				}
			}
		}

		return outImg;
	},
	backward: function(inImg, fW, fH, sX, sY, pX, pY) {
		// No need for an 'isLifted' check: there's only one possible lifted
		//    argument (inImg), so if we are backprop'ing, it must be lifted.
		var D = inImg.x.dims[0];
		var iH = inImg.x.dims[1];
		var iW = inImg.x.dims[2];
		var oH = this.x.dims[1];
		var oW = this.x.dims[2];
		var fN = fW*fH;

		for (var d = 0; d < D; d++) {
			var x = -pX;
			var y = -pY;
			for (var ay = 0; ay < oH; y += sY, ay++) {
				x = -pX;
				for (var ax = 0; ax < oW; x += sX, ax++) {
					var outidx = ax+oW*(ay+oH*d);
					var outDx = this.dx.data[outidx] / fN;
					for (var fy = 0; fy < fH; fy++) {
						var iy = y + fy;
						for (var fx = 0; fx < fW; fx++) {
							var ix = x + fx;
							if (iy>=0 && iy<iH && ix>=0 && ix<iW) {
								var inidx = ix+iW*(iy+iH*d);
								inImg.dx.data[inidx] += outDx;
							}
						}
					}
				}
			}
		}
	},
	getParents: function(inImg, fW, fH, sX, sY, pX, pY) {
		return ad.isLifted(inImg) ? [inImg] : [];
	}
});


MeanPoolNetwork.prototype.eval = function(img) {
	return meanpooling(img, this.filterWidth, this.filterHeight, this.strideX,
		this.strideY, this.padX, this.padY);
};


function meanpool(opts) {
	var fW = opts.filterWidth || opts.filterSize;
	var fH = opts.filterHeight || opts.filterSize;
	var sX = opts.strideX || opts.stride || fW;
	var sY = opts.strideY || opts.stride || fH;
	var pX = opts.padX || opts.pad || 0;
	var pY = opts.padY || opts.pad || 0;
	return new MeanPoolNetwork(fW, fH, sX, sY, pX, pY);
}


// Module exports -------------------------------------------------------------


module.exports = {
	maxpool: maxpool,
	minpool: minpool,
	meanpool: meanpool
};




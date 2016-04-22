'use strict';

function gaussianSample(mu, sigma) {
	var u, v, x, y, q;
	do {
		u = 1 - Math.random();
		v = 1.7156 * (Math.random() - 0.5);
		x = u - 0.449871;
		y = Math.abs(v) + 0.386595;
		q = x * x + y * (0.196 * y - 0.25472 * x);
	} while (q >= 0.27597 && (q > 0.27846 || v * v > -4 * u * u * Math.log(u)));
	return mu + sigma * v / u;
}

function deduplicate(list) {
	var retlist = [];
	for (var i = 0; i < list.length; i++) {
		var item = list[i];
		if (retlist.indexOf(item) === -1) {
			retlist.push(item);
		}
	}
	return retlist;
}

// source objects from which to copy are in arguments[1] - arguments[arguments.length-1]
function mergeObjects(tgt) {
	tgt = tgt || {};
	for (var i = 1; i < arguments.length; i++) {
		var src = arguments[i];
		for (var prop in src) {
			tgt[prop] = src[prop];
		}
	}
	return tgt;
}

function cloneObject(obj) {
	return mergeObjects({}, obj);
}

function mergeDefaults(obj, defaults) {
	return mergeObjects({}, defaults, obj);
}


module.exports = {
	gaussianSample: gaussianSample,
	deduplicate: deduplicate,
	mergeObjects: mergeObjects,
	cloneObject: cloneObject,
	mergeDefaults: mergeDefaults
};



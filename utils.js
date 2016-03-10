'use strict';

// Simple single-arg function memoization using stringified keys
function memoize(fn) {
	var cache = {};
	return function(x) {
		var key = x instanceof Function ? x.toString() : JSON.stringify(x);
		var y = cache[key];
		if (y === undefined) {
			y = fn(x);
			cache[key] = y;
		}
		return y;
	};
}

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


module.exports = {
	memoize: memoize,
	gaussianSample: gaussianSample,
	deduplicate: deduplicate
};
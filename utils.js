
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


module.exports = {
	memoize: memoize
};
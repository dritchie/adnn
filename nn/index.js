var Network = require('./network.js');
var lift = require('./lift.js');
var composition = require('./composition.js');
var liftedfns = require('./liftedfns');

module.exports = {
	Network: Network,
	lift: lift
};

var modules = [
	composition, liftedfns
];

for (var i = 0; i < modules.length; i++) {
	var m = modules[i];
	for (var prop in m) {
		module.exports[prop] = m[prop];
	}
}
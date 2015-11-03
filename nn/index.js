var Network = require('./network.js');
var lift = require('./lift.js');

module.exports = {
	Network: Network,
	lift: lift
};

// Include everything from composition
var composition = require('./composition.js');
for (var prop in composition) {
	module.exports[prop] = composition[prop];
}

// Networks
var lifted = require('./networks/lifted.js');
var fullyConnected = require('./networks/fullyConnected.js');
var modules = [
	lifted, fullyConnected
];
for (var i = 0; i < modules.length; i++) {
	var m = modules[i];
	for (var prop in m) {
		module.exports[prop] = m[prop];
	}
}
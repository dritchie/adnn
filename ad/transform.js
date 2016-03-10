'use strict';

var sweet = require('sweet.js');
var fs = require('fs');
var cp = require('child_process');

var adRequireStr = "var ad = require('adnn/ad');\n";
var macros = undefined;
function getMacros() {
	if (macros === undefined) {
		macros = sweet.loadNodeModule(__dirname, './macros.sjs');
	}
	return macros;
}

// Macro transform some code
function macroTransform(code) {
	var compiledCode = sweet.compile(code, {
		modules: getMacros(),
		readableNames: true
	}).code;
	return adRequireStr + compiledCode;
}

// Macro transform a module before requiring it.
// Must provide the full, resolved filename of the module.
// Optionally cache the transformed module code on disk.
// NOTE: Once cached, it won't be recompiled (unless you macroRequire the same
//    module again with 'cacheOnDisk' set to false). So this isn't like 'make',
//    which will detect changes and recompile for you. Thus, 'cacheOnDisk' is
//    best used for modules which aren't going to change (or which change very
//    infrequently).
function macroRequire(filename, cacheOnDisk) {
	var adfilename = filename + '.ad';
	var m = require.cache[adfilename];
	if (m !== undefined) {
		m = m.exports;
	} else {
		if (!cacheOnDisk || !fs.existsSync(adfilename)) {
			var code = fs.readFileSync(filename);
			var compiledCode = macroTransform(code);
			fs.writeFileSync(adfilename, compiledCode);
		}
		m = require(adfilename);
		if (!cacheOnDisk) {
			cp.execSync('rm -f ' + adfilename);
		}
	}
	return m;
}

module.exports = {
	macroTransform: macroTransform,
	macroRequire: macroRequire
}
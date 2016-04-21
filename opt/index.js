'use strict';


var utils = require('../utils.js');
var methods = require('./methods.js');
var optimize = require('./optimize.js');
var loss = require('./loss.js');


module.exports = utils.mergeObjects({}, methods, optimize, loss);
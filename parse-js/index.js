"use strict";
const esprima = require('esprima');

function parsePipeline(source) {
    const ast = esprima.parse(source);
    const tokens = esprima.tokenize(source, { loc: true });
    return { ast, tokens };
}

const fs = require('fs');

const [_node, _script, filename] = process.argv;

const source = fs.readFileSync(filename, 'utf8');
console.log(JSON.stringify(parsePipeline(source)));

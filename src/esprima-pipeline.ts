const esprima = require('esprima');

export interface Point {
  line: number;
  column: number;
};

export interface Location {
  start: Location;
  end: Location;
};

export interface Token {
  'type': string;
  value: string;
  loc: Location;
};

export interface PipelineResult {
  ast: Object;
  tokens: Token[];
};

/**
 * Returns {tokens, ast}.
 */
export default function parsePipeline(source: string): PipelineResult {
  const ast = esprima.parse(source);
  const tokens = esprima.tokenize(source, {loc: true});

  return {ast, tokens};
};

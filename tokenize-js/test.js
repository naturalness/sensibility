/*
 * Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import test from 'ava';

import {tokenize, checkSyntax} from './';

test('it tokenizes a trivial script', t => {
  const tokens = tokenize('$');

  t.is(1, tokens.length);
});

test('it returns line and column numbers', t => {
  const tokens = tokenize('\n$ ');
  t.is(1, tokens.length);
  const [token] = tokens;
  t.truthy(token.loc);
  t.truthy(token.loc.start);
  /* 1-indexed. */
  t.is(2, token.loc.start.line);
  /* 0-indexed. */
  t.is(0, token.loc.start.column);
});

test('it can deal with erroneous input', t => {
  let tokens;
  t.notThrows(() => {
    tokens = tokenize(`
      module.exports = function()
        console.log('Hello, world!');
      };
    `);
  });

  t.is(16, tokens.length);
});

test.skip('it can deal with illegal tokens', t => {
  let tokens;
  t.notThrows(() => {
    tokens = tokenize(`
      module.exports = function(# {

      };
    `);
  });

  t.is(10, tokens.length);
});

test('it can syntax check', t => {
  t.true(checkSyntax('function fun() { }'));
  t.false(checkSyntax('function fun() };'));
});

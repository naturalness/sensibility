#!/usr/bin/env node
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

'use strict';

const fs = require('fs');
const esprima = require('esprima');

module.exports.tokenize = tokenize;
module.exports.checkSyntax = checkSyntax;


if (require.main === module) {
  const source = fs.readFileSync('/dev/stdin', 'utf8');
  const shouldCheckSyntax =
    process.argv.slice(2).indexOf('--check-syntax') >= 0;
  if (shouldCheckSyntax) {
    process.exit(checkSyntax(source) ? 0 : 1);
  } else {
    console.log(JSON.stringify(tokenize(source)));
  }
}


function tokenize(source) {
  source = removeShebangLine(source);

  /* TODO: retry on illegal tokens. */

  const sourceType = deduceSourceType(source);
  const tokens = esprima.tokenize(source, {
    sourceType,
    loc: true,
    tolerant: true
  });

  return tokens;
}

function checkSyntax(source) {
  source = removeShebangLine(source);
  const sourceType = deduceSourceType(source);

  try {
    esprima.parse(source, { sourceType });
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Remove the shebang line, if there is one.
 */
function removeShebangLine(source) {
  return source.replace(/^#![^\r\n]+/, '');
}


/*
  Adapted from: http://esprima.org/demo/parse.js

  Copyright (C) 2013 Ariya Hidayat <ariya.hidayat@gmail.com>
  Copyright (C) 2012 Ariya Hidayat <ariya.hidayat@gmail.com>
  Copyright (C) 2011 Ariya Hidayat <ariya.hidayat@gmail.com>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */
function deduceSourceType(code) {
  try {
    esprima.parse(code, { sourceType: 'script' });
    return 'script';
  } catch (e) {
    return 'module';
  }
}

/* eslint no-console: 0 */

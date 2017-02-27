/*
 * Copyright 2016, 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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

CREATE TABLE source_file (
    hash    TEXT PRIMARY KEY NOT NULL, -- The SHA256 hash of the file
    owner   TEXT,
    repo    TEXT,
    path    TEXT NOT NULL,
    source  BLOB NOT NULL -- Stored as raw bytes; decode on use.

    -- I can't be assed to make this work:
    -- FOREIGN KEY (owner, repo) REFERENCES repository (owner, repo) ON DELETE CASCADE
);

CREATE TABLE parsed_source (
    hash    TEXT PRIMARY KEY,

    FOREIGN KEY (hash) REFERENCES source_file (hash) ON DELETE CASCADE
);

-- Insert three sources so I can actually test things. 
INSERT INTO source_file VALUES
    ('86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0',
     '<fake>', '<fake>',
     'source-a.js',
     CAST('(name) => console.log(`Hello, ${name}!`);' AS BLOB)),
    -- This one is minified.
    ('4ff5d5d6a8649d9227832cdd64e69b95b22eb8df9795793d795cd7bac32d57cc',
     '<fake>', '<fake>',
     'source-a.min.js',
     CAST('n=>console.log(`Hello, ${n}!`)' AS BLOB)),
    -- This one has a syntax error.
    ('dd8a0ec6ce3ac418859e7b5e15b8205d6b59ac2c1d673693b46a412e48e88be2',
     '<fake>', '<fake>',
     'broken.source-a.js',
     CAST('n=>console.log(`Hello, ${}!`)' AS BLOB)),
    ('3223cd0debcae2a1d23f2b265e0e61c75a2de8a3712de7b7068b93472a4bca91',
     '<fake>', '<fake>',
     'source-b.js',
     CAST('export default class Herp {};' AS BLOB));

-- Insert some sources so I can actually test things. 
INSERT INTO parsed_source VALUES
    ('86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0'),
    ('4ff5d5d6a8649d9227832cdd64e69b95b22eb8df9795793d795cd7bac32d57cc'),
    ('3223cd0debcae2a1d23f2b265e0e61c75a2de8a3712de7b7068b93472a4bca91');

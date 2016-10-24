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

-- Represents a source code repository.
CREATE TABLE repository (
    owner       TEXT NOT NULL, -- the owner of the repository
    repo        TEXT NOT NULL, -- the name of the repository
    license     TEXT, -- License of the file
    revision    TEXT, -- SHA of the latest revision

    PRIMARY KEY (repo, owner)
);

-- A single source file from a repository.
CREATE TABLE source_file (
    hash    TEXT PRIMARY KEY NOT NULL, -- The SHA256 hash of the file
    owner   TEXT,
    repo    TEXT,
    path    TEXT NOT NULL,
    source  BLOB NOT NULL, -- Stored as raw bytes; decode on use.

    FOREIGN KEY (owner, repo) REFERENCES repository (owner, repo) ON DELETE CASCADE
);

-- A file is inserted here if it has correct syntax, can be converted into an
-- AST and has a list of lexemes.
CREATE TABLE parsed_source (
    hash    TEXT PRIMARY KEY,
    ast     JSON NOT NULL, -- JSON
    tokens  JSON NOT NULL, -- JSON

    FOREIGN KEY (hash) REFERENCES source_file (hash) ON DELETE CASCADE
);

-- A file is inserted here if it has invalid syntax.
CREATE TABLE failure (
    hash    TEXT PRIMARY KEY
);

-- Define the list of approved licenses.
DROP TABLE IF EXISTS approved_license;
CREATE TABLE approved_license (
    name TEXT PRIMARY KEY NOT NULL
);
INSERT INTO approved_license (name) VALUES
    -- These are all permissive, but (may) require attribution.
    ('mit'), ('mpl-2.0'), ('unlicense'), ('bsd-2-clause'), ('isc'),
    ('apache-2.0'), ('cc0-1.0'), ('bsd-3-clause');

-- A repository is eligible if its license is approved.
CREATE VIEW IF NOT EXISTS eligible_repository AS
    SELECT *
      FROM repository
     WHERE license in (SELECT name FROM approved_license);

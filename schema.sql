/*
 * Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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
PRAGMA encoding = "UTF-8";
PRAGMA foreign_keys = ON;

-- It is recommended to use WAL mode and normal synchronization when updating
-- the database:
--  PRAGMA journal_mode = WAL;
--  PRAGMA synchronous = NORMAL;
-- It is recommend to use DELETE mode when accessing the database read-only.
--  PRAGMA journal_mode = DELETE;

-- Represents a source code repository.
CREATE TABLE repository (
    owner       TEXT NOT NULL,  -- the owner of the repository
    name        TEXT NOT NULL,  -- the name of the repository
    revision    TEXT NOT NULL,  -- SHA of the latest commit
    commit_date DATETIME NOT NULL, -- Timestamp of last commit
    license     TEXT,           -- License name of the file

    PRIMARY KEY (owner, name)
);

-- A source file from **any** repository.
CREATE TABLE source_file (
    hash    TEXT NOT NULL, -- The SHA256 hash of the file
    source  BLOB NOT NULL, -- Stored as raw bytes; decode on use.

    PRIMARY KEY (hash)
);

-- Relates a source file to repository.
CREATE TABLE repository_source(
    owner   TEXT NOT NULL,
    name    TEXT NOT NULL,
    hash    TEXT NOT NULL,
    path    TEXT NOT NULL, -- Path of file within this repository.

    FOREIGN KEY(owner, name) REFERENCES repository(owner, name)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY(hash) REFERENCES source_file(hash)
        ON DELETE CASCADE ON UPDATE CASCADE
);

-- Files with valid syntax are inserted here and embued with useful metadata.
CREATE TABLE source_summary (
    hash    TEXT PRIMARY KEY,
    sloc    INTEGER NOT NULL,   -- Source lines of code
    n_tokens INTEGER NOT NULL,  -- Number of tokens, total

    FOREIGN KEY(hash) REFERENCES source_file(hash)
        ON DELETE CASCADE ON UPDATE CASCADE
);

-- A file is inserted here if it has invalid syntax.
CREATE TABLE failure (
    hash    TEXT PRIMARY KEY,

    FOREIGN KEY(hash) REFERENCES source_file(hash)
        ON DELETE CASCADE ON UPDATE CASCADE
);

-- Represents a source code repository.
CREATE TABLE IF NOT EXISTS repository (
    owner       TEXT NOT NULL, -- the owner of the repository
    repo        TEXT NOT NULL, -- the name of the repository
    license     TEXT, -- License of the file
    revision    TEXT, -- SHA of the latest revision

    PRIMARY KEY (repo, owner)
);

-- A single source file from a repository.
CREATE TABLE IF NOT EXISTS source_file (
    hash    TEXT PRIMARY KEY NOT NULL, -- The SHA2 hash of the file
    owner   TEXT,
    repo    TEXT,
    path    TEXT NOT NULL,
    source  TEXT NOT NULL,

    FOREIGN KEY (owner, repo) REFERENCES repository (owner, repo) ON DELETE CASCADE
);

-- A file is inserted here if it has correct syntax, can be converted into an
-- AST and has a list of lexemes.
CREATE TABLE IF NOT EXISTS parse (
    hash    TEXT PRIMARY KEY,
    ast     TEXT NOT NULL, -- JSON
    lexemes TEXT NOT NULL, -- JSON

    FOREIGN KEY (hash) REFERENCES source_file (hash) ON DELETE CASCADE
);

-- A file is inserted here if it does not have valid syntax.
CREATE TABLE IF NOT EXISTS failure (
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

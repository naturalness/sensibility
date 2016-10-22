const fs = require('fs');
require('colors');

function queueTest() {
  console.log('Queue test'.green);

  const Q = require('./lib/q').default;
  const redis = require('redis');

  function Herp() {
  }
  Herp.prototype.toRedisString = function () {
    throw new Error('lol');
  };

  const client = redis.createClient();
  const q = new Q(client, 'herp');
  q.enqueue(new Herp)
    .then(n => {
      console.log(`Now there are ${n} items`)
      client.quit();
    })
    .catch(err => {
      console.error(err);
      client.quit();
    });
}

function parseTest() {
  console.log('Parse test'.green);
  const parsePipeline = require('./lib/esprima-pipeline.js').default;

  const file = fs.readFileSync('test.js', 'utf8');
  const result = parsePipeline(file);
  console.dir(result);
}

function dbTest() {
  console.log('Database test'.green);

  const sqlite3 = require('sqlite3').verbose();
  const db = new sqlite3.Database(':memory:');

  const schema = fs.readFileSync('./schema.sql', 'utf8');

  db.serialize(() => {
    db.run(schema);
  });

  db.close();
}

dbTest();

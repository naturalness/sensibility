const Q = require('./lib/q').default;
const redis = require('redis');
require('colors');

function queueTest() {
  console.log('Queue test'.green);

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
  const fs = require('fs');
  const parsePipeline = require('./lib/esprima-pipeline.js').default;

  const file = fs.readFileSync('test.js', 'utf8');
  const result = parsePipeline(file);
  console.dir(result);
}

parseTest();

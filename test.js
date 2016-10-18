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

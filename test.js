const Q = require('./lib/q').default;
const redis = require('redis');

const client = redis.createClient();
const q = new Q(client, 'herp');
q.enqueue('herp')
  .then((n) => console.log(`Now there are ${n} items`));

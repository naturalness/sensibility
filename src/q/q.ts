import {RedisClient} from 'redis';
import * as Promise from 'bluebird';

Promise.promisifyAll(RedisClient.prototype);
/* Monkey-patch the module. */
declare module 'redis' {
  interface RedisClient {
    lpushAsync(key: String, value: string): Promise<number>;
    rpoplpushAsync(src: string, dest: string): Promise<string>;
  }
}

/**
 * Object can be converted to a string representation in Redis.
 */
export interface RedisSerializable {
  toRedisString(): string;
}
/**
 * A Redis queue.
 */
export default class Q {
  constructor(public redis: RedisClient, public name: string) {
  }

  /**
   * Push an item to the queue.
   */
  enqueue(object: string | RedisSerializable): Promise<number> {
    const serialized =
      (typeof object === "string") ? object : object.toRedisString();
    return this.redis.lpushAsync(this.name, serialized);
  }

  /**
   * Atomically transfer an item from one queue to the other.
   */
  transfer(destination: string): Promise<string> {
    return this.redis.rpoplpushAsync(this.name, destination);
  }
}

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

import {RedisClient} from 'redis';
import * as bluebird from 'bluebird';

bluebird.promisifyAll(RedisClient.prototype);
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
    return Promise.resolve()
    .then(() => {
        const serialized =
          (typeof object === "string") ? object : object.toRedisString();

        return this.redis.lpushAsync(this.name, serialized);
    });
  }

  /**
   * Atomically transfer an item from one queue to the other.
   */
  transfer(destination: string): Promise<string> {
    return this.redis.rpoplpushAsync(this.name, destination);
  }
}

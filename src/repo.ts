/**
 * Uniquely identifies a GitHub repository.
 */
export default class RepositoryID {
  constructor(public owner: string, public repo: string) {
  }

  toRedisString() {
    return `${this.owner}/${this.repo}`;
  }

  /**
   * Convert to proper type from string.
   */
  static fromString(input: string): RepositoryID {
    let [owner, repo] = input.split('/');

    if (!(owner || repo)) {
      throw new Error(`Invalid repo string: ${input}`);
    }

    return new RepositoryID(owner, repo);
  }
}

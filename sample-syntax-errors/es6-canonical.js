const NO_MORE_DOCS = -1;

export class Documents {
  constructor(scorers) {
    this.last_doc = null;
    this.scorers = scorers;
  }

  doSomething() {
    for (var scorer in this.scorers) {
      if (scorer.nextDoc() === NO_MORE_DOCS) 
        this.last_doc = NO_MORE_DOCS;
        return;
      }
    }
  }

  toString() {
    return `Last doc: ${this.last_doc}`;
  }
}

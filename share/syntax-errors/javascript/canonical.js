var NO_MORE_DOCS = -1;

var Documents = module.exports = function Documents(scorers) {
  this.last_doc = null;
  this.scorers = scorers;
};

Documents.prototype.doSomething = function() {
  var scorer;
  for (scorer in this.scorers) {
    if (scorer.nextDoc() === NO_MORE_DOCS) 
      this.last_doc = NO_MORE_DOCS;
      return;
    }
  }
};

Documents.prototype.toString = function () {
  return `Last doc: ${this.last_doc}`;
};

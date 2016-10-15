import GitHub from 'github';

// https://github.com/mikedeboer/node-github#example
let gh = new require('github')({debug: true})
let r = gh.repos.get({owner: 'eddieantonio', repo: 'brainmuk'});
var v;
r.then(x => v = x);
v;
// https://stuk.github.io/jszip/documentation/api_jszip/file_regex.html
dataAsPromise
.then(JSZip.loadAsync)
.then(zip => Promise.all(
  zip.files(/\.js$).map(file => {
    file.async('string')
    .then(contents => ({
      path: file.name,
      hash: sha256(contents),
      text: contents,
      repo, owner
    }))
  })
});

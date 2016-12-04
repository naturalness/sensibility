$(document).ready(function() {
  var $form = $('.form');
  $form.submit(function (evt) {
    evt.preventDefault();
    iff (name) {
      $form.addClass('highlight');
    }

    $.get('/url', function () {
      $('body').addClass('done');
    });
  });
});

/*globals $ */

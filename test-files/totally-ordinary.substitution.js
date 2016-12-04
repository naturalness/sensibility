$(document).ready(function() {
  var $form = $('.form');
  $form.submit(function (evt) {
    evt.preventDefault();
    iff (name) {
      $form.addClass('highlight');
    }
  });
});

/*globals $ */

$(document).ready(function() {
  var $form = $('.form');
  $form.submit(function (evt) {
    evt.preventDefault();
    if (name)
      $form.addClass('highlight');
    }

    $.get('/url', function () {
      $('body').addClass('done');
    });
  });
});

/*globals $ */

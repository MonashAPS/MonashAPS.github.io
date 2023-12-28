$(document).ready(() => {

  hints = document.getElementsByClassName("button_unlock");
  for (var i = 0; i < hints.length; i++) {
    hints[i].onclick = function() {
      var cur = this;
      while (!cur.classList.contains("unlock")) {
        cur = cur.parentNode;
      }
      cur.classList.toggle("unlock");
    };
  }

});

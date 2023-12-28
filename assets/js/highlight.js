$(document).ready(() => {

  var all_gutters = document.getElementsByClassName("rouge-gutter");
  for (var j = 0; j < all_gutters.length; j++) {
    all_gutters[j].preserved = all_gutters[j].children[0].innerHTML.split("\n");
    all_gutters[j].cur_length = all_gutters[j].preserved.length;
    all_gutters[j].change_length = (s, l) => {
      s.cur_length = s.cur_length + l;
      s.children[0].innerHTML = s.preserved.slice(0, s.cur_length).join("\n");
    }
  }

  var all_code = document.getElementsByClassName("rouge-code");
  var fold_amounts = [];
  for (var j = 0; j < all_code.length; j++) {
      var pre = all_code[j].children[0];
      var lines = pre.innerHTML.split("\n");
      var open = true;
      var amount = 0;
      for (var k = 0; k<lines.length; k++) {
        amount ++;
          if (lines[k].includes("n")) {
              lines[k] = "<span class='highlight_backline_green'>" + lines[k].replace("n", "") + "</span>"
          }
          if (lines[k].includes("<span class=\"err\"></span><span class=\"n\">n</span>")) {
              lines[k] = "<span class='highlight_backline_green'>" + lines[k].replace("<span class=\"err\"></span><span class=\"n\">n</span>", "") + "</span>"
          }
          if (lines[k].includes("m")) {
              lines[k] = "<span class='highlight_backline_yellow'>" + lines[k].replace("m", "") + "</span>"
          }
          if (lines[k].includes("<span class=\"err\"></span><span class=\"n\">m</span>")) {
              lines[k] = "<span class='highlight_backline_yellow'>" + lines[k].replace("<span class=\"err\"></span><span class=\"n\">m</span>", "") + "</span>"
          }
          if (lines[k].includes("b")) {
            lines[k] = "<span class='highlight_backline_blue'>" + lines[k].replace("b", "") + "</span>"
          }
          if (lines[k].includes("<span class=\"err\"></span><span class=\"n\">b</span>")) {
              lines[k] = "<span class='highlight_backline_blue'>" + lines[k].replace("<span class=\"err\"></span><span class=\"n\">b</span>", "") + "</span>"
          }
          if (lines[k].includes("c")) {
              if (open) {
                  lines[k] = lines[k].replace("c", "<span class=\"code_fold code_fold_closed\"><button></button><span class=\"code_fold_span\">");
                  amount = 0;
              } else {
                  lines[k] = "</span></span>" + lines[k].replace("c", "");
                  pre.parentElement.parentElement.children[0].change_length(pre.parentElement.parentElement.children[0], -amount+1);
                  fold_amounts.push(amount-1);
              }
              open = !open;
            }
            if (lines[k].includes("<span class=\"err\"></span><span class=\"n\">c</span>")) {
                if (open) {
                    lines[k] = lines[k].replace("<span class=\"err\"></span><span class=\"n\">c</span>", "<span class=\"code_fold code_fold_closed\"><button></button><span class=\"code_fold_span\">");
                    amount = 0;
                } else {
                    lines[k-1] = lines[k-1] + "</span></span>"
                    lines[k] = lines[k].replace("<span class=\"err\"></span><span class=\"n\">c</span>", "");
                    pre.parentElement.parentElement.children[0].change_length(pre.parentElement.parentElement.children[0], -amount+1);
                    fold_amounts.push(amount-1);
                }
              open = !open;
          }
      }
      pre.innerHTML = lines.join("\n");
  }

  var all_folds = document.getElementsByClassName("code_fold");
  for (var j = 0; j < all_folds.length; j++) {
    var but = all_folds[j].children[0];
    but.click_index = j;
    but.onclick = (x) => {
      var pre = x.target.parentElement.parentElement.parentElement.parentElement.children[0];
      if (x.target.parentElement.classList.contains("code_fold_closed")) {
        pre.change_length(pre, fold_amounts[x.target.click_index]);
        x.target.parentElement.classList.remove("code_fold_closed");
        x.target.parentElement.classList.add("code_fold_opened");
      } else {
        pre.change_length(pre, -fold_amounts[x.target.click_index]);
        x.target.parentElement.classList.add("code_fold_closed");
        x.target.parentElement.classList.remove("code_fold_opened");
      }
    }
  }
});

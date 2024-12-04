var wideUnicodeRE = /[^\s\w\.,"'!?:;\-`=+!@#$%^&*\(\)\\\[\]\{\}<>\/|~]/;

jQuery.fn.extend({
insertAtCaret: function(myValue){
  return this.each(function(i) {
    if (document.selection) {
      //For browsers like Internet Explorer
      this.focus();
      var sel = document.selection.createRange();
      sel.text = myValue;
      this.focus();
    }
    else if (this.selectionStart || this.selectionStart == '0') {
      //For browsers like Firefox and Webkit based
      var startPos = this.selectionStart;
      var endPos = this.selectionEnd;
      var scrollTop = this.scrollTop;
      this.value = this.value.substring(0, startPos)+myValue+this.value.substring(endPos,this.value.length);
      this.focus();
      this.selectionStart = startPos + myValue.length;
      this.selectionEnd = startPos + myValue.length;
      this.scrollTop = scrollTop;
    } else {
      this.value += myValue;
      this.focus();
    }
  });
},
backspaceAtCaret: function(){
  return this.each(function(i) {    

    if (document.selection) 
    {
        this.focus();
        sel = document.selection.createRange();
        if(sel.text.length > 0)
        {
            sel.text="";
        }
        else
        {
            sel.moveStart("character",-1);
            sel.text="";
        }
        sel.select();
    }
    else if (this.selectionStart || this.selectionStart == "0")
    { 
        var startPos = this.selectionStart;
        var endPos = this.selectionEnd;

        // For everything but spaces, assume wide Unicode characters
        // (TODO: Is there a way to detect whether the char is wide?)
        var deletedString = this.value.substring(startPos-1, endPos);
        //alert("Deleted string: " + deletedString);
        if (wideUnicodeRE.test(deletedString))
            startPos -= 1;

        this.value = this.value.substring(0, startPos-1) + this.value.substring(endPos, this.value.length);
        this.selectionStart = startPos-1;
        this.selectionEnd = startPos-1;
        this.focus();
    } 
    else 
    {
        this.value=this.value.substr(0,(this.value.length-1));
        this.focus();
    }
  })
}
});

keyboard = $('#keyboard');

var html = '<div id="keyboard-letters">';
for (i=66560; i <= 66599; ++i) {
	var keyName = "deseret-key-" + i;
	html += '<div id="' + keyName + '" class="deseret-key" title="' + i + '"><div class="deseret-char">&#' + i + ';</div><div class="keyboard-char"></div></div>';
}
html += '<div id="deseret-key-comma" class="deseret-key"><div class="deseret-char">,</div></div>';
html += '<div id="deseret-key-period" class="deseret-key"><div class="deseret-char">.</div></div>';
html += '<div id="deseret-key-question" class="deseret-key"><div class="deseret-char">?</div></div>';
html += '<div id="deseret-key-bang" class="deseret-key"><div class="deseret-char">!</div></div>';
html += '<div id="keyboard-bottom-row"></div><div id="deseret-key-shift" class="deseret-key" title="Shift">Shift</div>';
html +=   '<div id="deseret-key-space" class="deseret-key" title="Space"><div class="deseret-char">&nbsp;</div></div>';
html += '</div><div id="keyboard-delete-column"><div id="deseret-key-backspace" class="deseret-key" title="Backspace">Delete</div></div>';
keyboard.html(html);

$('.deseret-key').click(function() {
    if (this.id=='deseret-key-shift') {
        $(this).toggleClass('on');
        update_keyboard();
    }
	else if (this.id == 'deseret-key-backspace') {
		// JavaScript is UCS-2 and doesn't handle wide characters; so to delete 
		// a Deseret Alphabet Unicode character we actually have to do two deletes.
		// It would be nice to detect whether the character at the caret is wide or not,
		// but for now we assume that it is.
		$('#output_well').backspaceAtCaret(); //.backspaceAtCaret();
	}
	else {
        var deseretChar = $(this).children('.deseret-char').text();
        if( ! $('#deseret-key-shift').hasClass('on') ) {
            deseretChar = deseretChar.toLowerCase();
        }
		$('#output_well').insertAtCaret(deseretChar);
	}

    var scope = angular.element($("#output_well")).scope();
    scope.$apply(function() {
        scope.deseret = $('#output_well').val();
        scope.deseretToEnglish();
    });
});

$.getJSON("/js/deseret_names.json", function(json) {
    for (i=66560; i <= 66599; ++i) {
        var keyId = "#deseret-key-" + i;
        var keyDef = json[i.toString()];
        var letterName = keyDef['name'];
        var keyMap = keyDef['ascii'];
        var key = $(keyId);
        key.attr("title", letterName);
        key.attr("lower-key", keyMap[0]);
        key.attr("upper-key", keyMap[1])
    }
    update_keyboard();
});

function update_keyboard() {
    var isShiftOn = $('#deseret-key-shift').hasClass('on');
    for (i=66560; i <= 66599; ++i) {
        var keyId = "#deseret-key-" + i;
        var key = $(keyId);
        var deseretCharDiv = key.children('.deseret-char');
        var keyboardCharDiv = key.children('.keyboard-char');

        if (isShiftOn) {
            deseretCharDiv.text(deseretCharDiv.text().toUpperCase());
            keyboardCharDiv.text(key.attr("upper-key"));
        }
        else {
            deseretCharDiv.text(deseretCharDiv.text().toLowerCase());
            keyboardCharDiv.text(key.attr("lower-key"));
        }
    }
}

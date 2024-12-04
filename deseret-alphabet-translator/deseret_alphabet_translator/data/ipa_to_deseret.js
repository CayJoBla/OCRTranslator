// Taken from Neil Alexander Walker, "A Complete Guide to Reading and Writing the Deseret Alphabet"
// http://copper.chem.ucla.edu/~jericks/Historical%20or%20Technical/Linguistics/Deseret_Guide.pdf
//
// To strip comments and create valid JSON file:
//
//   $ python -m jsmin ipa_to_deseret.js > ipa_to_deseret.json

// Missing: 66586, 66587, 665
{
    "i":	"66560",	/* sounds like the "e" in "see" */
    "eI":	"66561",	/* sounds like the "a" in "day" */
    "A":	"66562",	/* sounds like the "a" in "far" */
    "O":	"66563",	/* sounds like the "o" in "dog" */
    "oU":	"66564",	/* sounds like the "o" in "boat" */
    "u":	"66565",	/* sounds like the "oo" in "too" */
    "I":	"66566",	/* sounds like the "i" in "hid" */
    "(@)":	"66567",	/* sounds like the "a" in "air" */
    "E":	"66567",	/* sounds like the "e" in "red" */
	"&":	"66568",	/* sounds like the "a" in "dab" */
    "A":	"66569",	/* sounds like the "o" in "bob" */
    "-":	"66570",	/* no perfect equivalent for schwa in Deseret Alphabet -- see 2.2.5 in Walker, "A Complete Guide to Reading and Writing the Deseret Alphabet" -- sounds like the "ir" glide in "tire" or the  "dl" glide in "handle" or the "den" glide in "sodden" (dipthong little schwa) */
	"@":	"66570",	/* sounds like the "a" in "ado" or the glide "e" in "system" (dipthong schwa) */
    "[@]":	"66570",	/* sounds like the "u" in "cup" */
    "@r":	"66571",	/* sounds like the "u" in "burn" */
    "U":	"66571",	/* sounds like the "oo" in "book" */
	"aI":	"66572",	/* sounds like the "i" in "ice" */
    "AU":	"66573",	/* sounds like the "ow" in "how" */
    "hw":	"66574",	/* sounds like the "w" in "white" */
    "w":	"66574",	/* sounds like the "w" in "win" */
    "j":	"66575",	/* sounds like the "y" in "you" */
    "h":	"66576",	/* sounds like the "h" in "had" */
    "p":	"66577",	/* sounds like the "p" in "imp" */
	"b":	"66578", 	/* sounds like the "b" in "nab" */
    "t":	"66579",	/* sounds like the "t" in "tap" */
    "d":	"66580",	/* sounds like the "d" in "pod" */
	"tS":	"66581",	/* sounds like the "ch" in "ouch" */
	"dZ":	"66582",	/* sounds like the "g" in "vegetably" */
	"k":	"66583",	/* sounds like the "c" in "act" */
    "g":	"66584",	/* sounds like the "g" in "fig" */
	"f":	"66585",	/* sounds like the "f" in "elf" */
    "v":	"66586",	/* sounds like the "v" in "average" */
    "T":	"66587",	/* sounds like the "th" in "bath" */
    "D":	"66588",	/* sounds like the "th" in "the" */
    "s":	"66589",	/* sounds like the "s" in "sip" */
    "z":	"66590",	/* sounds like the "z" in "zoo" */
    "S":	"66591",	/* sounds like the "sh" in "she" */
    "Z":	"66592",	/* sounds like the "s" in "vision" */
    "r":	"66593",	/* sounds like the "r" in "ire" */
	"l":	"66594",	/* sounds like the "l" in "ail" */
	"m":	"66595",	/* sounds like the "m" in "aim" */
    "n":	"66596",	/* sounds like the "n" in "and" */
	"N":	"66597",	/* sounds like the "ng" in "bang" */
	"Oi":	"66598",	/* sounds like the "oi" in "oil" */       // later removed
//  "u":    "66599",    /* sounds like the "ew" in "drew" */      // not sure how this differs from 66565
}


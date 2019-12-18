#target photoshop
// links
// http://lacuisine.tech/2018/08/09/execute-python-script-from-photoshop/
// https://www.davidebarranca.com/2012/10/scriptui-window-in-photoshop-palette-vs-dialog/
// https://www.makeuseof.com/tag/how-to-automate-photoshop-with-scripts/-

// https://community.adobe.com/t5/Photoshop/Interactive-ScriptUI-Builder/td-p/10481030
current_document = app.activeDocument;
current_layer = current_document.activeLayer;
var clCopy = current_document.activeLayer.copy();

// create temporary doc with selected layer from current document
docHeight = current_document.height;
docWidth = current_document.width;
docResolution = current_document.resolution;
$.writeln("docHeight", docHeight);
tempDoc = app.documents.add( docWidth,docHeight, docResolution, "tempDoc", NewDocumentMode.RGB);
app.activeDocument = tempDoc;
tempDoc.paste();
app.activeDocument = current_document;

outPath = "/Users/n/School/MED3/P3/Code/P3-G306/STRUCTURE/PHOTOSHOP/PhotoshopOut.jpg"
fileRef = File(outPath)

docExportOptions = new ExportOptionsSaveForWeb

docExportOptions.format = SaveDocumentType.PNG //-24 //JPEG, COMPUSERVEGIF, PNG-8, BMP
docExportOptions.transparency = true
docExportOptions.blur = 0.0
docExportOptions.includeProfile = false
docExportOptions.interlaced = false
docExportOptions.optimized = true
docExportOptions.quality = 100
docExportOptions.PNG8 = false

current_document.exportDocument(fileRef, ExportType.SAVEFORWEB, docExportOptions)

var execFilePath = "/Users/n/School/MED3/P3/Code/P3-G306/STRUCTURE/PHOTOSHOP/c.command"

// we create the command file, and setup it before opening it
var execFile = new File(execFilePath);
execFile.encoding = "UTF-8";
execFile.lineFeed = "Unix";
execFile.open("w");

// now we write on the .command file,
// that it's a bash script, calling python with an argument
 execFile.write(
   "#!/bin/bash\n\n" +
   "python /Users/n/School/MED3/P3/Code/P3-G306/STRUCTURE/PHOTOSHOP/function.py " + outPath + " 'SOME OTHER TEST ARG'");

// you can repeat that command and write many lines instead

// when the file is written, we simply execute it
File(execFilePath).execute();


// https://www.joshwright.com/tips/photoshop-scripting-user-interface
var dlg = new Window("dialog{text:'Script Interface',bounds:[100,100,561,269],\
iwtfkhamhc:EditText{bounds:[16,16,444.95,94] , text:'Your text goes here' ,properties:{multiline:false,noecho:false,readonly:false}},\
button0:Button{bounds:[17,102,117,122] , text:'Save' },\
button1:Button{bounds:[236,101,336,121] , text:'Cancel' },\
button2:Button{bounds:[345,101,445,121] , text:'Whatever' },\
slider0:Slider{bounds:[18,138,173,148] , minvalue:0,maxvalue:100,value:0},\
checkbox0:Checkbox{bounds:[190,133,261,154] , text:'Checkbox Text' },\
dropdown0:DropDownList{bounds:[299,134,443,149],properties:{items:['Select One']}}\
};");

// https://www.davidebarranca.com/2012/10/scriptui-window-in-photoshop-palette-vs-dialog/
dlg.button0.onClick = function() {
  $.writeln("save clicked!");
};
dlg.show();

function W() {
  var w = new Window('dialog', 'Title');
  var b = w.add('button', undefined, 'OK', {
    name: 'ok'
  });
  return w
}
$.setTimeout = function(func, time) {
  $.sleep(time);
  func();
};

$.setTimeout(function() {
  alert("hello world")
}, 1000);
// current_document.// OPTIMIZE:
// var w = W();
// $.writeln(w.show());

maskPath = File("/Users/n/School/MED3/P3/Code/P3-G306/STRUCTURE/PHOTOSHOP/PythonOut.jpeg")

var openedMask = open(maskPath)

openedMask.artLayers[0].duplicate(current_document);
openedMask.close();
// app.activeDocument = current_layer;

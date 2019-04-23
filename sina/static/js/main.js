/*let app = new annotator.App();
app.include(annotator.ui.main, {
    element: document.querySelectorAll('#sina-sentence')[0], // getElement methods do not work, nor jQuery $
    editorExtensions: [annotator.ui.tags.editorExtension],
    viewerExtensions: [
        annotator.ui.tags.viewerExtension
    ]
});
app.include(annotator.storage.http, {
    prefix: 'http://127.0.0.1:5000/api' //should include on page to read in here
});
app.start().then(function () {
     app.annotations.load();
});*/

//CVN annotation library
Annotesto.init({container:'sina-sentence'})

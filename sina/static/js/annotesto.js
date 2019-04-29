(function(window){
    'use strict';
    
    const _init = (
	{
	    storageUrl = 'http://127.0.0.1:5000',
	    container = '',
	    eventContainer = '',
	    annotElementTag = 'annot-8',
	    legend = true,
	    startColor = null, // should be ]0-1] to have same selection of colors
	    preloadTags = true
	}
    )=>{
	let body = document.getElementsByTagName("body")[0];
	let annotContainer = container ?
	    document.getElementById(container):body;
	let AnnotElement = document.registerElement(annotElementTag);
	let AnnotElementRe = new RegExp('<\/?'+annotElementTag+'.*?>', 'g');
	let eventRegion = eventContainer ?
	    document.getElementById(eventContainer):body;
	let annotationsMade = 0;
	window.Annotesto.config = Object(null);
	window.Annotesto.config.annotElementTag = annotElementTag;
	window.Annotesto.pristineText = annotContainer.innerText;
	window.Annotesto.pristineHTML = annotContainer.innerHTML.split(AnnotElementRe).join('');
	window.Annotesto.annotContainer = annotContainer;

	// Legend and storage options
	window.Annotesto.legend = legend ? createLegend(startColor):false;
	window.Annotesto.storage = storageUrl ? new Storage(storageUrl,preloadTags):false;

	// Initialise annotation documents
	window.Annotesto.doc = new ADoc(
	    annotContainer.getAttribute('data-adoc-id'),
	    annotContainer
	);

	// Even initialisation
	eventRegion.addEventListener("mouseup",()=>{
            let selection = window.getSelection();
	    if (selection.isCollapsed) return;
	    let range = selection.getRangeAt(0);
	    if (fitSelection(range, annotContainer)){
		let annotation = new Annotation(
		    range,
		    annotationsMade++,
		    window.Annotesto.doc,
		    new AnnotElement()
		);
		annotation.tag();
		range.collapse();
		if (annotation.tag_annotations) {
		    annotation.spanSelection();
		    window.Annotesto.annotations.push(
			annotation
		    );
		}
	    }
	});

	// Finished init
	console.log("Annotator Initialised");
    }
    
    window.Annotesto = {
	init:_init,
	annotations: [],
	fitSelection: fitSelection
    }
})(window);

class ADoc { // Annotation document - could be one per page or more
    constructor(id,adocElement,init_tags) { //adocElement -> will put a button before
	this.type = 'doc_annotation';
	this.origin = document.baseURI;
	this.id = id;
	this.adocElement = adocElement;
	this.doc_annotations = init_tags?init_tags:'';
	this.createButton();
	
	// If connected storage retrieve document annotation
	if (window.Annotesto.storage)
	    window.Annotesto.storage.fetchDocAnnotation(this.id)
	    .then(response => this.button.innerText = response['doc_tags']);
    }

    createButton() {
	this.button = document.createElement('button');
	this.button.id = 'adoc-' + this.id;
	this.button.innerText = this.doc_annotations;
	this.adocElement.parentNode.insertBefore(
	    this.button,
	    this.adocElement
	);
	this.button.addEventListener('click', e => {
	    let prevAnnot = this.doc_annotations;
	    this.doc_annotations = prompt('Document tags (separate with ,):', this.doc_annotations);
	    this.button.innerText = this.doc_annotations;
	    if (prevAnnot !== this.doc_annotations && window.Annotesto.storage) window.Annotesto.storage.sendAnnotation(this);
	});
    }
}

class Annotation {
    constructor(range,id,annotParentDoc,annotElement,init_tags) {
	this.type = 'segment_annotation';
	this.origin = document.baseURI;
	this.originalRange = range.cloneRange();
	this.label = range.toString();
	this.id = id;
	this.parent_id = annotParentDoc.id;
	this.previousTextMatches();
	this.annotElement = annotElement; //document.createElement('span');
	if (!init_tags) { // First time annotation
	    this.annotElement.setAttribute('data-id', id);
	    this.annotElement.className = 'annot-8hl';
	} else { // Previously annotated tags have been provided
	    this.tag_annotations = init_tags.join(', ');
	    this.annotElement.className = 'annot-8hl,annot-prev';
	    this.annotElement.style.backgroundColor = window.Annotesto.legend ?
		window.Annotesto.legend(this.tag_annotations):'rgba(255,255,0,0.3)';
	    this.prevContent = document.createDocumentFragment();
	    this.prevContent.append(annotElement.innerHTML);
	}
	this.annotElement.addEventListener("click",()=>{
	    this.tag();
	});
    }

    get range() {
	if (this.originalRange) return this.originalRange;
	else {
	    let range = document.createRange();
	    range.setStartBefore(this.annotElement);
	    range.setEndAfter(this.annotElement);
	    return range;
	}
    }
    
    tag() {
	let prevTags = this.tag_annotations;
	let wasTagged = Boolean(prevTags);
	let tags = prompt('Tag (separate with ,):', this.tag_annotations);
	this.tag_annotations = tags;
	if (tags) {
	    this.annotElement.style.backgroundColor = Annotesto.legend(tags);
	    if (window.Annotesto.storage && wasTagged && (prevTags !== tags))
		window.Annotesto.storage.updateAnnotation(this);
	    else if (window.Annotesto.storage) window.Annotesto.storage.sendAnnotation(this)
		.then(data => console.log(JSON.stringify(data)))
		.catch(error => console.error(error));
	} else this.deleteAnnotation();
    }

    spanSelection() {
	this.prevContent = this.originalRange.cloneContents();
	this.originalRange.surroundContents(this.annotElement);
	this.originalRange = false;
    }

    deleteAnnotation() {
	if (window.Annotesto.storage)
	    // Could delete html annotation only after storage confirmation
	    // but interactively might look weird
	    window.Annotesto.storage.deleteAnnotation(this);
	let range = this.range;
	range.extractContents(); // cannot use this.range after extracting element
	range.insertNode(this.prevContent);
	window.Annotesto.annotations.splice(
	    Annotesto.annotations.indexOf(this),1
	);
    }

    previousTextMatches() { //should be run at initialisation only as dom-mods can mess up range
	let count = 0;
	let before = this.range.cloneRange();
	before.collapse(true); //move end of range to beginning
	before.setStart(window.Annotesto.annotContainer,0);
	let beforeText = before.toString()
	for (let i=0; i=beforeText.indexOf(this.label,i)+1;) {
	    count++;
	    //console.log(i);
	}
	this.precedingMatches = count
    }

}

class Storage{
    constructor(url,loadLegendTags) {
	this.url= url.replace(/\/$/,"")+'/api';
	this.fetchAnnotations();
	if (loadLegendTags) this.fetchTags();
    }

    sendAnnotation(annotation) {
	return fetch(this.url+'/annotations', {
	    //Info: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
            method: "POST", 
            mode: "cors",
            cache: "no-cache",
            credentials: "same-origin",
            headers: {
		"Content-Type": "application/json",
            },
            redirect: "follow", // manual, *follow, error
            body: JSON.stringify(annotation),
	})
	    .then(response => response.json());
    }

    fetchAnnotations(){
	let annotels = document.getElementsByTagName(window.Annotesto.config.annotElementTag);
	// TODO first loop over annotation documents
	let docid = window.Annotesto
	    .annotContainer
	    .getAttribute('data-adoc-id'); //window.Annotesto.doc.id;
	for (let i=0; i<annotels.length; i++) {
	    let dataid = annotels[i].getAttribute('data-id')
	    console.log('Retrieving annotation '+dataid);
	    fetch(this.url+`/search/${docid}/${dataid}`, {
		method: "GET", 
		mode: "cors",
		cache: "no-cache",
		credentials: "same-origin",
		headers: {
		    "Content-Type": "application/json",
		},
	    })
		.then(response => response.json()) // response.json creates another promise
		.then(response => {
		    //console.log(response);
		    let range = document.createRange();
		    range.setStartBefore(annotels[i]);
		    range.setEndAfter(annotels[i]);
		    window.Annotesto.annotations.push(
			new Annotation(
			    range,
			    dataid,
			    window.Annotesto.doc,//TODO In the future get out of response parent_id
			    annotels[i],
			    response['tags']
			)
		    );
		});
	}
    }

    fetchDocAnnotation(doc_id) {
	return fetch(this.url+`/docannotation/${doc_id}`, {
            method: "GET", 
            mode: "cors",
            credentials: "same-origin",
            headers: {
		"Content-Type": "application/json",
            },
	})
	    .then(response => response.json())
    }
    
    updateAnnotation(annotation) {
	return fetch(this.url+'/update', {
            method: "POST", 
            mode: "cors",
            credentials: "same-origin",
            headers: {
		"Content-Type": "application/json",
            },
            body: JSON.stringify(annotation),
	})
	    .then(response => response.json())
	    .then(response => console.log(response));
    }
    
    deleteAnnotation(annotation) {
	return fetch(this.url+'/delete', {
            method: "DELETE", 
            mode: "cors",
            credentials: "same-origin",
            headers: {
		"Content-Type": "application/json",
            },
            body: JSON.stringify(annotation),
	})
	    .then(response => response.json())
	    .then(response => console.log(response));
    }

    fetchTags() {
	return fetch(this.url+'/tags', {
            method: "GET", 
            mode: "cors",
            credentials: "same-origin",
            headers: {
		"Content-Type": "application/json",
            },
	})
	    .then(response => response.json())
	    .then(response => response.forEach(a => window.Annotesto.legend(a)))
    }
}

// Functions
function fitSelection(range,fitDomElement) {
    let fitRange = document.createRange();
    fitRange.selectNode(fitDomElement)
    if (
	// If a selection is made entirely outside of the annotation container, false is returned
	(fitRange.compareBoundaryPoints(0,range) === 1 && fitRange.compareBoundaryPoints(3,range) === 1) ||
	(fitRange.compareBoundaryPoints(1,range) === -1 && fitRange.compareBoundaryPoints(2,range) === -1)
    ) return false;
    // Only if at least start or end point falls within the annotation container, is the selection
    // aligned appropriately to the annotation container
    if (fitRange.compareBoundaryPoints(0,range) === 1) range.setStartBefore(fitDomElement);
    if (fitRange.compareBoundaryPoints(2,range) === -1) range.setEndAfter(fitDomElement);
    return true;
}

function createLegend(startcolor) {
    let legend = document.createElement('div');
    let tags = Object(null);
    legend.id = 'annot8legend';
    window.Annotesto.annotContainer.parentNode.insertBefore(
	legend,
	window.Annotesto.annotContainer.nextSibling
    );
    let title = document.createElement('h3');
    title.innerText = 'Legend:'
    legend.appendChild(title);
    let table = document.createElement('table');
    legend.appendChild(table);
    let randomColorPicker = randomColorSelector(startcolor);
    
    function createRow(tag,color,colorText) {
	if (tag in tags) return tags[tag];
	else {
	    let row = document.createElement('tr');
	    let col1 = document.createElement('td');
	    let col2 = document.createElement('td');
	    col1.innerText = tag;
	    if (colorText) col2.innerText = colorText;
	    color = color ? color:randomColorPicker();
	    col2.style.backgroundColor = color ? color:randomColorPicker();
	    row.appendChild(col1);
	    row.appendChild(col2);
	    legend.appendChild(row);
	    tags[tag] = color;
	    return color;
	}
    }
    createRow('Tag','','Color');
    return createRow;
}

function randomColorSelector(startcolor) {
    /* References:
       https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
       https://stackoverflow.com/questions/17242144/javascript-convert-hsb-hsv-color-to-rgb-accurately
     */
    let golden_ratio_conjugate = 0.618033988749895;
    let h = startcolor ? startcolor:Math.random()
    function pickColor() {
	h += golden_ratio_conjugate;
	h %= 1;
	return HSVtoRGB(h, 0.5, 0.95);
    }
    return pickColor;
}

function HSVtoRGB(h, s, v) {
    var r, g, b, i, f, p, q, t;
    if (arguments.length === 1) {
        s = h.s, v = h.v, h = h.h;
    }
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    return `rgba(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)},0.3)`;
}

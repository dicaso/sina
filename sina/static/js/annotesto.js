(function(window){
    'use strict';
    
    const _init = (
	{
	    // Annotesto configuration settings
	    storageUrl = 'http://127.0.0.1:5000',
	    container = '', // id name if single container, class name for multiple
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
	let annotContainers = null;
	if (!annotContainer) // if null should be described by class
	    annotContainers = document.getElementsByClassName(container);
	let AnnotElement = document.registerElement(annotElementTag);
	let AnnotElementRe = new RegExp('<\/?'+annotElementTag+'.*?>', 'g');
	let eventRegion = eventContainer ?
	    document.getElementById(eventContainer):body;
	window.Annotesto.singleDoc = Boolean(annotContainer);
	window.Annotesto.annotationsMade = 0;
	window.Annotesto.config = Object(null);
	window.Annotesto.config.annotElementTag = annotElementTag;
	/*window.Annotesto.pristineText = annotContainer.innerText;
	window.Annotesto.pristineHTML = annotContainer.innerHTML.split(AnnotElementRe).join('');*/
	if (annotContainer) window.Annotesto.annotContainer = annotContainer;
	else window.Annotesto.annotContainers = annotContainers;

	// Legend and storage options
	window.Annotesto.legend = legend ? createLegend(startColor):false;
	window.Annotesto.storage = storageUrl ? new Storage(storageUrl,preloadTags):false;

	// Initialise annotation documents
	if (annotContainer) window.Annotesto.doc = new ADoc(
	    annotContainer.getAttribute('data-adoc-id'),
	    annotContainer
	);
	else {
	    window.Annotesto.docs = [];
	    for (let i=0; i<annotContainers.length; i++)
		window.Annotesto.docs.push(
		    new ADoc(
			annotContainers[i].getAttribute('data-adoc-id'),
			annotContainers[i]
		    )
		);
	}
	
	// Event initialisation
	eventRegion.addEventListener("mouseup",()=>{
            let selection = window.getSelection();
	    if (selection.isCollapsed) return;
	    let range = selection.getRangeAt(0);
	    let fitDomElement = null;
	    if (fitDomElement = fitSelection(range, annotContainer)){
		
		let annotation = new Annotation(
		    range,
		    window.Annotesto.annotationsMade++,
		    window.Annotesto.singleDoc ?
			window.Annotesto.doc:
			window.Annotesto.docs[Number(fitDomElement.getAttribute('data-adoc-id'))],
		    new AnnotElement()
		);
		range.collapse();
		if (annotation.tag_annotations) {
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
	this.id = Number(id);
	this.parent_gid = Number(adocElement.getAttribute('data-adoc-gid'));
	this.adocElement = adocElement;
	this.doc_annotations = init_tags?init_tags:'';
	this.createButton();
	
	// If connected storage retrieve document annotation
	if (window.Annotesto.storage)
	    window.Annotesto.storage.fetchDocAnnotation(this.id,this.parent_gid)
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
	// Parent id needs to match with 0-indexed ADoc location
	if (window.Annotesto.annotContainers)
	    this.parent_gid = annotParentDoc.parent_gid;
	this.previousTextMatches();
	this.annotElement = annotElement; //document.createElement('span');
	if (!init_tags) { // First time annotation
	    this.annotElement.setAttribute('data-id', id);
	    this.annotElement.className = 'annot-8hl';
	    this.tag()
	    try {if (this.tag_annotations) this.spanSelection();}
	    catch (error) {
		if (window.Annotesto.storage)
		    window.Annotesto.storage.deleteAnnotation(this);
		console.log('Partially overlapping annotations not implemented!');
	    }

	} else { // Previously annotated tags have been provided
	    this.tag_annotations = init_tags.join(', ');
	    this.annotElement.className = 'annot-8hl annot-prev';
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
	} else if (wasTagged) this.deleteAnnotation();
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
	before.setStart(
	    window.Annotesto.annotContainer?
		window.Annotesto.annotContainer:window.Annotesto.annotContainers[this.parent_id],
	    0
	);
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
	if (loadLegendTags) this.fetchTags();
	this.fetchAnnotations();
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
	for (let i=0; i<annotels.length; i++) {
	    let docid = annotels[i].getAttribute('data-docid');
	    let pageid = annotels[i].getAttribute('data-gid'); // not relevant for singleDoc
	    let dataid = Number(annotels[i].getAttribute('data-id'));
	    window.Annotesto.annotationsMade = Math.max(
		window.Annotesto.annotationsMade,
		dataid+1
	    );
	    console.log('Retrieving annotation '+dataid);
	    let url = window.Annotesto.singleDoc?
		this.url+`/search/${docid}/${dataid}`:
		this.url+`/search/${pageid}/${docid}/${dataid}`;
	    fetch(url, {
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
			    window.Annotesto.singleDoc?
				window.Annotesto.doc:window.Annotesto.docs[docid],
			    annotels[i],
			    response['tags']
			)
		    );
		});
	}
    }

    fetchDocAnnotation(doc_id,page_id) {
	let url = window.Annotesto.singleDoc?
	    this.url+`/docannotation/${doc_id}`:
	    this.url+`/docannotation/${page_id}/${doc_id}`;
	return fetch(url, {
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
    let fitRange = null;
    if (fitDomElement) { // Only one annotContainer
	fitRange = document.createRange();
	fitRange.selectNode(fitDomElement);
	if (
	    // If a selection is made entirely outside of the annotation container, false is returned
	    (fitRange.compareBoundaryPoints(0,range) === 1 && fitRange.compareBoundaryPoints(3,range) === 1) ||
		(fitRange.compareBoundaryPoints(1,range) === -1 && fitRange.compareBoundaryPoints(2,range) === -1)
	) return false;
    } else { // Multiple possibilities, select first that has some overlap
	let overlapFound = false;
	for (let i=0; i<window.Annotesto.annotContainers.length; i++) {
	    fitDomElement = window.Annotesto.annotContainers[i];
	    fitRange = document.createRange();
	    fitRange.selectNode(fitDomElement);
	    if (
		overlapFound = !((fitRange.compareBoundaryPoints(0,range) === 1 && fitRange.compareBoundaryPoints(3,range) === 1) ||
				(fitRange.compareBoundaryPoints(1,range) === -1 && fitRange.compareBoundaryPoints(2,range) === -1))
	    ) break;
	}
	if (!overlapFound) return false;
    }
    
    // Only if at least start or end point falls within the annotation container, is the selection
    // aligned appropriately to the annotation container
    if (fitRange.compareBoundaryPoints(0,range) === 1) range.setStart(fitDomElement,0);
    if (fitRange.compareBoundaryPoints(2,range) === -1) range.setEnd(fitDomElement,fitDomElement.childNodes.length);
    //setStartBefore setEndAfter => do not work as they mess up the spanning of the new element
    return fitDomElement;
}

function createLegend(startcolor) {
    let legend = document.createElement('div');
    let tags = Object(null);
    legend.id = 'annot8legend';

    // Add legend to DOM
    if (window.Annotesto.annotContainer)
	window.Annotesto.annotContainer.parentNode.insertBefore(
	    legend,
	    window.Annotesto.annotContainer.nextSibling
	);
    else {
	window.Annotesto.annotContainers[0].parentNode.insertBefore(
	    legend,
	    window.Annotesto.annotContainers[0]
	);
	window.Annotesto.annotContainers[0].parentNode.insertBefore(
	    document.createElement('br'),
	    window.Annotesto.annotContainers[0]
	);
    }

    // Create legend elements
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

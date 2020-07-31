async function predict(model, image) {
	
	let tensor = tf.browser.fromPixels(image, 3)
		.resizeNearestNeighbor([224, 224]) // change the image size
		.expandDims()
		.toFloat()
		.reverse(-1); // RGB -> BGR
		
	let predictions = await model.predict(tensor).data();
	
	console.log(predictions);
	
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 2);	
		
	return top5;
}


async function predict_imagemfixa(model, imageID) {
	let image = $('#' + imageID).get(0);
	let list = $("#" + imageID + "_pred");
	
	let top5 = await predict(model, image);

	list.empty();
	top5.forEach(function (p) {
		list.append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
	});	
}

async function predict_camera(model, image) {
	let list = $("#prediction-list");
	
	let top5 = await predict(model, image);

	list.empty();
	top5.forEach(function (p) {
		if (p.probability > 0.5)
			list.append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
	});	
}

$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let model;
let imagens = ["nota2","nota5","nota10","nota20","nota50","nota100"]
$( document ).ready(async function () {
	$('.progress-bar').show();
    console.log( "Loading model..." );
    model = await tf.loadGraphModel('model/model.json');
    console.log( "Model loaded." );
	$('.progress-bar').hide();
});

$("#predict-button").click(async function () {
	for (i in imagens)
		predict_imagemfixa(model, imagens[i]);
});

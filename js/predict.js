TARGET_CLASSES = {
	0: "10_1_back",
	1: "10_1_front",
	2: "10_2_back",
	3: "10_2_front",
	4: "100_1_front",
	5: "100_2_back",
	6: "100_2_front",
	7: "2_1_back",
	8: "2_1_front",
	9: "2_2_back",
	10: "2_2_front",
	11: "20_1_back",
	12: "20_1_front",
	13: "20_2_back",
	14: "20_2_front",
	15: "5_1_back",
	16: "5_1_front",
	17: "5_2_back",
	18: "5_2_front",
	19: "50_1_back",
	20: "50_1_front",
	21: "50_2_back",
	22: "50_2_front",
	23: "none"
};

async function load_model() {
    console.log( "Loading model..." );
    model = await tf.loadGraphModel('model/model.json');
    console.log( "Model loaded." );	
}

async function predict(model, image) {
	
	let tensor = tf.browser.fromPixels(image, 3)
		.resizeNearestNeighbor([224, 224]) // change the image size
		.expandDims()
		.toFloat()
		.reverse(-1); // RGB -> BGR
		
	let predictions = await model.predict(tensor).data();
	
	// console.log(predictions);
	
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

async function predict_online(canvas) {

	canvas.toBlob(function(blob){

		let xhr = new XMLHttpRequest();
		xhr.open('POST', 'https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/15a5a098-47fe-4b9a-8443-67a62ea7e205/classify/iterations/Iteration1/image');
		xhr.setRequestHeader("Content-Type", "application/octet-stream");
		xhr.setRequestHeader("Prediction-Key", '35e4bf2c77a04d8ca756a27156393477');
		xhr.send(blob);
	
		xhr.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				predictions = JSON.parse(this.responseText).predictions;

				predictions.sort(function (a, b) {
					return b.probability - a.probability;
				});

				console.log(predictions);
				
				predictions.forEach(function (p) {
					if (p.probability > 0.1)
						list.append(`<li>Online model: ${p.tagName}: ${p.probability.toFixed(6)}</li>`);
				});	
			}
		};

	},'image/jpg');	

}

let pred_anterior, preds;

async function predict_local(model, image) {
	// list.empty();
	// list.append(`<li>Custom Vision Model - Local</li>`);
	// list.append(`<li>------------------------------------</li>`);

	let top5 = await predict(model, image);

	predClass = top5[0].className;
	predProba = top5[0].probability.toFixed(2);

	if (predClass != 'none' & predProba > 0.1) {
		if (predClass == pred_anterior)
			preds++;
		else
			preds = 0;
	}


	if (preds > 10) {
		list.append(`<li>${predClass}: ${predProba}</li>`);
		preds = 0;
	}
		
	pred_anterior = predClass;	

	// top5.forEach(function (p) {
	// 	if (p.probability > 0.1)
	// 		list.append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
	// });	
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

$("#predict-button").click(async function () {
	for (i in imagens)
		predict_imagemfixa(model, imagens[i]);
});

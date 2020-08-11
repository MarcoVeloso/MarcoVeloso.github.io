let TARGET_CLASSES = {
	0: "100_g2_frente",
	1: "100_g2_verso",
	2: "2_g2_frente",
	3: "2_g2_verso",
	4: "5_g2_frente",
	5: "5_g2_verso"
};

async function load_model() {
	console.log( "Loading model..." );
    model = await tf.loadGraphModel('model/model.json');
    console.log( "Model loaded." );	
}

async function load_tesseract() {
    const { createWorker } = Tesseract;
    worker = createWorker({
      workerPath: 'lib/tesseract/worker.min.js',
      langPath: 'lib/tesseract',
      corePath: 'lib/tesseract/tesseract-core.wasm.js',
    });
    await worker.load();
    await worker.loadLanguage('eng');
    await worker.initialize('eng');
}

function print_predicts(target_class, probability, threshold=0.1){
	if (probability >= threshold)
		list.append(`<li>${target_class}: ${probability.toFixed(3)}</li>`);
		//list.append(`<li>${target_class}</li>`);
	else
		list.append(`<li>Indefinido</li>`);
}

function preprocess(image, obj_detect=false) {
	let input_size = 224;

	if (obj_detect)
		input_size = 416;

	let tensor = tf.browser.fromPixels(image, 3)
		.resizeNearestNeighbor([input_size, input_size]) 
		.expandDims()
		.toFloat()
		.reverse(-1); 

	return tensor;
}

async function postprocess(outputs) {
	function logistic(x) {
		if (x > 0) {
		    return (1 / (1 + Math.exp(-x)));
		} else {
		    const e = Math.exp(x);
		    return e / (1 + e);
		}
	}

	const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17];
	const num_anchor = ANCHORS.length / 2;
	const channels = outputs[0][0][0].length;
	const height = outputs[0].length;
	const width = outputs[0][0].length;

	const num_class = channels / num_anchor - 5;

	let boxes = [];
	let scores = [];
	let classes = [];

	for (var grid_y = 0; grid_y < height; grid_y++) {
		for (var grid_x = 0; grid_x < width; grid_x++) {
			let offset = 0;

			for (var i = 0; i < num_anchor; i++) {
				let x = (logistic(outputs[0][grid_y][grid_x][offset++]) + grid_x) / width;
				let y = (logistic(outputs[0][grid_y][grid_x][offset++]) + grid_y) / height;
				let w = Math.exp(outputs[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2] / width;
				let h = Math.exp(outputs[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2 + 1] / height;

				let objectness = tf.scalar(logistic(outputs[0][grid_y][grid_x][offset++]));
				let class_probabilities = tf.tensor1d(outputs[0][grid_y][grid_x].slice(offset, offset + num_class)).softmax();
				offset += num_class;

				class_probabilities = class_probabilities.mul(objectness);
				let max_index = class_probabilities.argMax();

				boxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
				scores.push(class_probabilities.max().dataSync()[0]);
				classes.push(max_index.dataSync()[0]);
			}
		}
	}

	boxes = tf.tensor2d(boxes);
	scores = tf.tensor1d(scores);
	classes = tf.tensor1d(classes);

	const selected_indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10);
	return [await boxes.gather(selected_indices).array(), await scores.gather(selected_indices).array(), await classes.gather(selected_indices).array()];
}

async function predict(model, image) {
	let tensor = preprocess(image, true);
		
	let predictions = await model.predict(tensor);

	console.log(predictions);

	return predictions;
	
	// pred = await postprocess(await predictions.array());
	
	
	// let top5 = Array.from(predictions)
	// 	.map(function (p, i) { // this is Array.map
	// 		return {
	// 			probability: p,
	// 			className: TARGET_CLASSES[i] // we are selecting the value from the obj
	// 		};
	// 	}).sort(function (a, b) {
	// 		return b.probability - a.probability;
	// 	}).slice(0, 2);	
		
	// return top5;
}

async function predict_online(canvas) {

	canvas.toBlob(function(blob){

		let xhr = new XMLHttpRequest();
		xhr.open('POST', 'https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/d723b35b-ce99-4003-9d3e-6e442c9eb020/classify/iterations/Iteration8/image');
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

async function predict_ocr(image) {
	list.empty();

	await predict_ocr_local(image);
	await predict_ocr_online();
}

async function predict_ocr_online() {

	canvas.toBlob(function(blob){

        $.ajax({
			url: 'https://southcentralus.api.cognitive.microsoft.com/vision/v3.0/read/analyze',
            type: "POST",
			data: blob,
			contentType: false,
			processData: false,
			headers: {
				"Content-Type":"application/octet-stream",
				"Ocp-Apim-Subscription-Key":"35e4bf2c77a04d8ca756a27156393477"
			},

			success: function(result,status,xhr){
				var operationLocation = xhr.getResponseHeader("Operation-Location");

				setTimeout(function () {

					$.ajax({
						url: operationLocation,
						type: "GET",
						headers: {
							"Content-Type":"application/json",
							"Ocp-Apim-Subscription-Key":"35e4bf2c77a04d8ca756a27156393477"
						},
			
						success: function(result){
							data = result.analyzeResult.readResults[0].lines;

							list.append(`<li>------------------------------</li>`)
							list.append(`<li>OCR Online (Microsoft Vision)</li>`);

							for (let i in data)
								list.append(`<li>${data[i].text}</li>`);
						},
					});

				}, 3000);

			},
		});
	},'image/jpg');	

}

async function predict_ocr_local(image) {
	list.append(`<li>OCR Local (Tesseract.js)</li>`);
	
	const { data: { text } } = await worker.recognize(image);
	
	list.append(`<li>${text}</li>`);
	
	$("#responseTextArea").val(text);
}

async function predict_local(model, image) {
	let top5 = await predict(model, image);

	top5.forEach(function (p) {
		if (p.probability > 0.1)
			list.append(`<li>Local model: ${p.className}: ${p.probability.toFixed(6)}</li>`);
	});	
}

async function predict_obj_detect(model, image) {
	list.empty();
	list.append(`<li>Reconhecimento da nota em andamento...</li>`);

	let predictions = await predict(model, image);

	preds = await postprocess(await predictions.array());	

	list.empty();

	console.log(preds);

	if (preds[1][0] > 0.1) {
		if (preds[2][0] == preds[2][1])
			list.append(`<li>${TARGET_CLASSES[preds[2][0]]}</li>`);
		else
			list.append(`<li>Indefinido</li>`);
	}
	else
		list.append(`<li>Indefinido</li>`);	

	// for (pred in preds[1])
	// 	print_predicts(TARGET_CLASSES[preds[2][pred]], preds[1][pred]);

}


import { FMnistData } from './fashion-data.js';
var canvas, ctx, saveButton, clearButton;
var pos = { x: 0, y: 0 };
var rawImage;
var model;

function getModel() {
    model = tf.sequential();

    // Convolutional layers
    model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 64, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 80, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 40, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    // Compile the model
    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(),
        metrics: ['accuracy']
    });

    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];

    const container = { name: 'Model Training', styles: { height: '1000px' } };

    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 6000;
    const TEST_DATA_SIZE = 1000;

    // Get the training batches and resize them
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE); // Use nextTrainBatch for training data
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), // Corrected to TRAIN_DATA_SIZE
            d.labels
        ];
    });

    // Get the testing batches and resize them
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

function setPosition(e) {
    pos.x = e.clientX - 100;
    pos.y = e.clientY - 100;
}

function draw(e) {
    if (e.buttons != 1) return;
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}

function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
}

async function save() {
    var raw = tf.browser.fromPixels(rawImage, 1);
    var resized = tf.image.resizeBilinear(raw, [28, 28]);
    var tensor = resized.expandDims(0);

    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();

    var classNames = ["T-shirt/top", "Trouser", "Pullover",
        "Dress", "Coat", "Sandal", "Shirt",
        "Sneaker", "Bag", "Ankle boot"];

    alert(classNames[pIndex]);
}

function init() {
   
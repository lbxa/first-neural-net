import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const site = path.join(root, 'website');
const notebooks = path.join(root, 'manual/jupyter');
const assets = path.join(site, 'src/assets/archive');
const downloads = path.join(site, 'public/archive');
await Promise.all([mkdir(assets, { recursive: true }), mkdir(downloads, { recursive: true }), mkdir(path.join(site, 'src/data'), { recursive: true })]);

const training = JSON.parse(await readFile(path.join(notebooks, 'training.ipynb'), 'utf8'));
const architecture = JSON.parse(await readFile(path.join(notebooks, 'architecture.ipynb'), 'utf8'));
const text = (value) => Array.isArray(value) ? value.join('') : value;
function outputFor(book, source) {
  const cell = book.cells.find((cell) => text(cell.source).trim() === source);
  if (!cell) throw new Error(`Missing archived cell: ${source}`);
  return cell.outputs.map((output) => text(output.text ?? output.data?.['text/plain'] ?? '')).join('\n');
}
function numbers(output) {
  return [...output.matchAll(/-?\d+\.\d+(?:e[-+]?\d+)?/gi)].map(([value]) => Number(value));
}

const predictions = numbers(outputFor(training, 'NN.forward(train_x)')).map((value) => value * 100);
const targets = numbers(outputFor(training, 'train_y')).map((value) => value * 100);
const initialPredictions = numbers(outputFor(architecture, 'NN.forward(x_train)')).map((value) => value * 100);
if (predictions.length !== 4 || targets.length !== 4) throw new Error('Expected four archived predictions and targets.');
const errors = predictions.map((value, i) => value - targets[i]);
const mse = errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length;
const result = {
  predictions, targets, initialPredictions,
  numericalGradient: numbers(outputFor(training, 'numerical_gradient')),
  analyticGradient: numbers(outputFor(training, 'computed_gradient')),
  optimizerOutput: outputFor(training, 'T = Trainer(NN)\nT.train(train_x, train_y, test_x, test_y)').trim(),
  metrics: { mae: errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length, rmse: Math.sqrt(mse), mse, dataLoss: mse / 20000 },
};
await writeFile(path.join(site, 'src/data/archive.json'), JSON.stringify(result, null, 2) + '\n');

const plot = training.cells.flatMap((cell) => cell.outputs ?? []).find((output) => output.data?.['image/png']);
if (!plot) throw new Error('The original training plot is missing.');
await writeFile(path.join(assets, 'training-cost.png'), Buffer.from(text(plot.data['image/png']), 'base64'));
for (const name of ['neuron.png', 'a_neuron.png', 'ann_model.png']) {
  await copyFile(path.join(notebooks, 'img', name), path.join(assets, name));
}
for (const name of ['intro.ipynb', 'architecture.ipynb', 'training.ipynb']) {
  await copyFile(path.join(notebooks, name), path.join(downloads, name));
}
await copyFile(path.join(root, 'src/feed_forward_net.py'), path.join(downloads, 'feed_forward_net.py'));
await copyFile(path.join(root, 'src/feed_forward_net.py'), path.join(site, 'src/data/original.py'));
console.log('Prepared original figures, notebooks, source, and metrics from saved notebook outputs.');

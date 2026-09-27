import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const diagram = await readFile(new URL('../src/assets/diagrams/feedforward-network.svg', import.meta.url));
const backdrop = await readFile(new URL('../src/assets/social-preview.svg', import.meta.url));
const { data, info } = await sharp(diagram, { density: 216 })
  .resize({ width: 1080, height: 300, fit: 'inside' })
  .png()
  .toBuffer({ resolveWithObject: true });
await sharp(backdrop)
  .composite([{ input: data, left: Math.round((1200 - info.width) / 2), top: 245 }])
  .png({ compressionLevel: 9 })
  .toFile(fileURLToPath(new URL('../public/social-preview.png', import.meta.url)));
console.log('Prepared the 1200×630 social preview from the TikZ diagram.');

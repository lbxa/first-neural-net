import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const source = new URL('../public/diagrams/feedforward-network.tex', import.meta.url);
const destination = new URL('../src/assets/diagrams/feedforward-network.svg', import.meta.url);
const fingerprint = createHash('sha256')
  .update(await readFile(source))
  .update(await readFile(fileURLToPath(import.meta.url)))
  .digest('hex');
const marker = `data-tikz-build="${fingerprint}"`;
let existing = '';
try {
  existing = await readFile(destination, 'utf8');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}

// Keep the generated vector in Git so unchanged builds need no TeX installation.
if (!process.argv.includes('--force') && existing.includes(marker)) {
  console.log('TikZ network diagram is up to date.');
} else {
  const directory = await mkdtemp(path.join(tmpdir(), 'feedforward-tikz-'));
  try {
    execFileSync('latex', [
      '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error',
      `-output-directory=${directory}`, fileURLToPath(source),
    ], { cwd: path.dirname(fileURLToPath(source)), stdio: 'pipe' });
    const svgPath = path.join(directory, 'feedforward-network.svg');
    execFileSync('dvisvgm', [
      '--no-fonts', '--exact-bbox', '--bbox=preview', '--optimize',
      '--precision=4', '--currentcolor', `--output=${svgPath}`,
      path.join(directory, 'feedforward-network.dvi'),
    ], { stdio: 'pipe' });
    const svg = (await readFile(svgPath, 'utf8')).replace('<svg ', `<svg ${marker} `);
    if (!svg.includes(marker) || !svg.includes('viewBox=')) {
      throw new Error('The TikZ compiler did not produce a valid scalable SVG.');
    }
    await mkdir(new URL('../src/assets/diagrams/', import.meta.url), { recursive: true });
    await writeFile(destination, svg);
    console.log('Compiled the TikZ network diagram to inline SVG.');
  } catch (error) {
    if (error.code === 'ENOENT') {
      throw new Error('To regenerate the diagram, install TeX Live or MacTeX with TikZ, standalone, and dvisvgm on PATH. Commit the resulting SVG with the TeX source.', { cause: error });
    }
    if (error.stdout) console.error(error.stdout.toString());
    if (error.stderr) console.error(error.stderr.toString());
    throw error;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

// Fontes Mermaid são sempre renderizadas por Pele, inclusive para visualização no GitHub.
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { render } from 'pele';

const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(import.meta.url);
const { version } = require('pele/package.json');
const check = process.argv.includes('--check');

function staticColors(svg) {
  // Exportação estática: alguns visualizadores de SVG não resolvem custom properties CSS.
  // Resolve os aliases/fallbacks emitidos pelo Pele, preservando seu layout e suas cores.
  const variables = new Map([...svg.matchAll(/(--[\w-]+):([^;"}]+)/g)].map((m) => [m[1], m[2]]));
  for (let depth = 0; svg.includes('var(') && depth < 20; depth++) {
    svg = svg.replace(/var\((--[\w-]+)(?:,([^()]*))?\)/g, (_, name, fallback) => {
      const value = variables.get(name) ?? fallback;
      if (value === undefined) throw new Error(`Variável CSS sem valor: ${name}`);
      return value;
    });
  }
  if (svg.includes('var(')) throw new Error('Variáveis CSS cíclicas ou não resolvidas.');
  return svg;
}

async function sources(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = await Promise.all(entries.map(async (entry) => {
    const name = path.join(directory, entry.name);
    return entry.isDirectory() ? sources(name) : name.endsWith('.mmd') ? [name] : [];
  }));
  return paths.flat().sort();
}

let failures = 0;
for (const sourcePath of await sources(path.join(root, 'docs'))) {
  const relative = path.relative(path.join(root, 'docs'), sourcePath);
  const target = path.join(root, 'docs', 'diagrams', relative.replace(/\.mmd$/, '.svg'));
  const source = await readFile(sourcePath, 'utf8');
  const id = createHash('sha256').update(relative.split(path.sep).join('/')).digest('hex').slice(0, 12);
  const result = render(source, {
    idPrefix: `rag-${id}`,
    fontFamily: 'Arial, sans-serif',
    fontSize: 16,
    padding: 24,
    responsive: false,
    autoDirection: false,
    links: false,
    images: false,
  });
  // Fundo explícito torna o SVG legível ao ser aberto como imagem no GitHub em tema escuro.
  const svg = staticColors(result.svg).replace(
    /(<svg\b[^>]*>)/, '$1<rect width="100%" height="100%" fill="#ffffff"/>',
  );
  const output = `<!-- Gerado com Pele ${version}; fonte: docs/${relative.split(path.sep).join('/')} -->\n${svg}\n`;
  if (check) {
    const actual = await readFile(target, 'utf8').catch(() => '');
    if (actual !== output) {
      console.error(`SVG ausente/desatualizado: ${path.relative(root, target)}. Execute npm run docs:diagrams.`);
      failures++;
    }
  } else {
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, output);
    console.log(`${path.relative(root, sourcePath)} → ${path.relative(root, target)} (Pele ${version})`);
  }
}
if (failures) process.exitCode = 1;

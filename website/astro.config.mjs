// @ts-check
import { defineConfig } from 'astro/config';
import { createRequire } from 'node:module';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import accessibleMath from './scripts/accessible-math.mjs';

// KaTeX's HTML positioning and stylesheet must come from the same release.
const require = createRequire(import.meta.url);
const stylesheetVersion = require('katex').version;
const rendererVersion = createRequire(require.resolve('rehype-katex'))('katex').version;
if (stylesheetVersion !== rendererVersion) {
  throw new Error('KaTeX version mismatch: align the direct katex dependency with rehype-katex before building.');
}

const site = process.env.SITE_URL ?? 'https://mlp.lbxa.net';
const url = new URL(site);
if (!['http:', 'https:'].includes(url.protocol) || url.pathname !== '/' || url.search || url.hash || url.username || url.password) {
  throw new Error('SITE_URL must be an absolute HTTP(S) origin without a path, credentials, query, or fragment.');
}
export default defineConfig({
  site,
  output: 'static',
  trailingSlash: 'always',
  integrations: [mdx(), sitemap()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: 'error', throwOnError: true }], accessibleMath],
    }),
    shikiConfig: { theme: 'github-light-high-contrast', wrap: false },
  },
  vite: { plugins: [tailwindcss()] },
});

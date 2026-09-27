import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { load } from 'cheerio';
import sharp from 'sharp';

const args = process.argv.slice(2);
const directory = path.resolve(args.find((arg) => !arg.startsWith('--')) ?? 'dist');
const routes = ['/', '/foundations/', '/data/', '/architecture/', '/training/', '/results/', '/source/'];
const documents = new Map();
for (const route of routes) {
  const filename = path.join(directory, route.slice(1), 'index.html');
  documents.set(route, load(await readFile(filename, 'utf8')));
}
const home = documents.get('/');
const origin = home('link[rel="canonical"]').attr('href');
if (args.includes('--require-site')) assert(origin, 'Publishable builds require SITE_URL.');
if (process.env.SITE_URL) assert.equal(origin, new URL(process.env.SITE_URL).href);
const base = origin ?? 'https://seo-check.invalid/';
const titles = new Set();
const descriptions = new Set();
let checkedLinks = 0;

for (const [route, $] of documents) {
  const title = $('title').text();
  const description = $('meta[name="description"]').attr('content');
  assert.equal($('title').length, 1, `${route}: one page title`);
  assert.equal($('meta[name="description"]').length, 1, `${route}: one description`);
  assert(title && !titles.has(title), `${route}: unique, nonempty title`);
  assert(description && !descriptions.has(description), `${route}: unique, nonempty description`);
  titles.add(title);
  descriptions.add(description);
  assert.equal($('html').attr('lang'), 'en');
  assert.equal($('h1').length, 1, `${route}: one main heading`);
  assert(!$('meta[name="robots"]').attr('content')?.includes('noindex'), `${route}: indexable chapter`);
  assert.equal($('script:not([type="application/ld+json"])').length, 0, `${route}: no executable scripts`);
  assert.equal($('meta[property="og:description"]').attr('content'), description);
  assert.equal($('meta[name="twitter:card"]').attr('content'), 'summary_large_image');

  if (origin) {
    const canonical = new URL(route, origin).href;
    assert.equal($('link[rel="canonical"]').length, 1);
    assert.equal($('link[rel="canonical"]').attr('href'), canonical);
    assert.equal($('meta[property="og:url"]').attr('content'), canonical);
    assert.equal($('meta[property="og:image"]').attr('content'), new URL('/social-preview.png', origin).href);
    assert.equal($('meta[name="twitter:image"]').attr('content'), $('meta[property="og:image"]').attr('content'));
    assert.equal($('meta[property="og:image:width"]').attr('content'), '1200');
    assert.equal($('meta[property="og:image:height"]').attr('content'), '630');
    assert($('meta[property="og:image:alt"]').attr('content'));
    assert.equal($('script[type="application/ld+json"]').length, 1);
    const schema = JSON.parse($('script[type="application/ld+json"]').text());
    assert.equal(schema['@context'], 'https://schema.org');
    const graph = schema['@graph'];
    const article = graph.find((item) => item['@type'] === 'Article');
    const person = graph.find((item) => item['@type'] === 'Person');
    assert.equal(article.url, canonical);
    assert.equal(article.headline, $('meta[property="og:title"]').attr('content'));
    assert.equal(article.description, description);
    assert.equal(article.author['@id'], person['@id']);
    assert.equal(person.name, $('meta[name="author"]').attr('content'));
    assert.equal(graph.find((item) => item['@type'] === 'WebPage').url, canonical);
    assert.equal(graph.find((item) => item['@type'] === 'WebSite').url, origin);
    if (route !== '/') {
      const crumbs = graph.find((item) => item['@type'] === 'BreadcrumbList').itemListElement;
      assert.deepEqual(crumbs.map((item) => item.position), [1, 2]);
      assert.equal(crumbs[0].item, origin);
      assert.equal(crumbs[1].item, canonical);
      assert.equal(crumbs[1].name, $('h1').text());
    }
  } else {
    assert.equal($('link[rel="canonical"], meta[property="og:url"], script[type="application/ld+json"]').length, 0, 'No invented production URLs in local builds.');
  }

  for (const element of $('a[href], img[src], link[rel="stylesheet"]').toArray()) {
    const href = $(element).attr('href') ?? $(element).attr('src');
    const url = new URL(href, new URL(route, base));
    if (url.origin !== new URL(base).origin) continue;
    const target = documents.get(url.pathname);
    if (target) {
      if (url.hash) {
        const id = decodeURIComponent(url.hash.slice(1));
        assert(target('[id]').toArray().some((node) => target(node).attr('id') === id), `${route}: missing anchor ${href}`);
      }
    } else {
      await access(path.join(directory, decodeURIComponent(url.pathname).slice(1)));
    }
    checkedLinks++;
  }
  console.log(`${route} ${title.length}-character title; ${description.length}-character description; metadata and links pass.`);
}

const notFound = load(await readFile(path.join(directory, '404.html'), 'utf8'));
assert(notFound('meta[name="robots"]').attr('content')?.includes('noindex'));
assert.equal(notFound('link[rel="canonical"], script[type="application/ld+json"]').length, 0);
const image = await sharp(path.join(directory, 'social-preview.png')).metadata();
assert.equal(image.format, 'png');
assert.equal(image.width, 1200);
assert.equal(image.height, 630);
const robots = await readFile(path.join(directory, 'robots.txt'), 'utf8');
assert(robots.includes('Allow: /'));
assert(!/^Disallow:\s*\/\s*$/m.test(robots));
if (origin) {
  assert(robots.includes(`Sitemap: ${new URL('/sitemap-index.xml', origin).href}`));
  const index = load(await readFile(path.join(directory, 'sitemap-index.xml'), 'utf8'), { xml: true });
  const locations = [];
  for (const node of index('sitemap > loc').toArray()) {
    const url = new URL(index(node).text());
    assert.equal(url.origin, new URL(origin).origin);
    const sitemap = load(await readFile(path.join(directory, url.pathname.slice(1)), 'utf8'), { xml: true });
    locations.push(...sitemap('url > loc').toArray().map((item) => sitemap(item).text()));
  }
  assert.deepEqual(locations.sort(), routes.map((route) => new URL(route, origin).href).sort(), 'Sitemap contains exactly the canonical chapters.');
} else {
  assert(!robots.includes('Sitemap:'));
  console.warn('Local SEO checks pass. Set SITE_URL and rerun with --require-site before publishing.');
}
console.log(`Verified ${documents.size} chapters, ${checkedLinks} internal links/assets, the 404, social image, and crawler metadata.`);

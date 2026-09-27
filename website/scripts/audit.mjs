import { mkdir, writeFile } from 'node:fs/promises';
import lighthouse from 'lighthouse';
import desktopConfig from 'lighthouse/core/config/desktop-config.js';
import { launch } from 'chrome-launcher';

const origin = process.env.AUDIT_URL ?? 'http://127.0.0.1:4323';
const routes = ['/', '/foundations/', '/data/', '/architecture/', '/training/', '/results/', '/source/'];
const output = new URL('../reports/lighthouse/', import.meta.url);
await mkdir(output, { recursive: true });
const chrome = await launch({ chromeFlags: ['--headless', '--disable-dev-shm-usage'] });
const summary = [];
try {
  for (const device of ['mobile', 'desktop']) {
    for (const route of routes) {
      const options = {
        port: chrome.port,
        logLevel: 'error',
        output: ['html', 'json'],
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
      };
      const result = await lighthouse(new URL(route, origin).href, options, device === 'desktop' ? desktopConfig : undefined);
      if (!result || result.lhr.runtimeError) {
        throw new Error(JSON.stringify(result?.lhr.runtimeError ?? 'No Lighthouse result'));
      }
      const name = device + '-' + (route.replaceAll('/', '') || 'overview');
      await writeFile(new URL(name + '.html', output), result.report[0]);
      await writeFile(new URL(name + '.json', output), result.report[1]);
      const entry = {
        route, device, fetchedAt: result.lhr.fetchTime,
        scores: Object.fromEntries(Object.entries(result.lhr.categories).map(([key, value]) => [key, Math.round(value.score * 100)])),
        metrics: {
          fcp: result.lhr.audits['first-contentful-paint'].numericValue,
          lcp: result.lhr.audits['largest-contentful-paint'].numericValue,
          cls: result.lhr.audits['cumulative-layout-shift'].numericValue,
        },
        failures: Object.entries(result.lhr.audits).filter(([, audit]) => audit.score !== null && audit.score < 1 && audit.scoreDisplayMode !== 'informative').map(([id, audit]) => ({ id, score: audit.score, title: audit.title })),
      };
      summary.push(entry);
      await writeFile(new URL('summary.json', output), JSON.stringify(summary, null, 2) + '\n');
      console.log(JSON.stringify(entry));
    }
  }
} finally {
  chrome.kill();
}

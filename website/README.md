# Feedforward Neural Network for Regression

A static Astro research archive for Lucas Barbosa's 2017 project. Seven chapters cover the problem, original data, architecture, backpropagation, saved results, and complete source.

## Development

Run commands from this directory. Requires Node 22.12+ and Bun.

~~~sh
bun install
bun run dev --port 4322
~~~

The dev command prepares the original archive and starts Astro in background mode. Manage it with:

~~~sh
bun astro dev status
bun astro dev logs
bun astro dev stop
~~~

The site expects the parent repository's manual/ and src/ directories. Keep website/ inside the original repository. Original code and notebooks are never modified.

## Production

~~~sh
bun run check
bun run build
bun run check:seo --require-site
bun run preview --host 127.0.0.1 --port 4323
~~~

The production origin is **https://mlp.lbxa.net**, the default in astro.config.mjs. To build for a different public origin, override SITE_URL:

~~~sh
SITE_URL=https://research.example.org bun run build
bun run check:seo --require-site
~~~

Replace the example with the alternate domain. Set SITE_URL in the hosting provider's build environment or export it in the shell; .env.example documents the value but is not loaded by astro.config.mjs. SITE_URL must be an HTTP(S) origin without a subpath, credentials, query, or fragment. The site uses root-relative links and expects deployment at the domain root.

Every normal build emits canonical URLs, Open Graph URLs, absolute social image URLs, a seven-chapter sitemap, the sitemap reference in robots.txt, and JSON-LD for the configured production origin. `check:seo --require-site` rejects a build missing those deployment settings. The final domain does not depend on the local preview URL.

## Cloudflare Worker deployment

Run these commands from website/. Wrangler is installed as a development dependency and configured in wrangler.jsonc. Authenticate on a new machine, then validate and deploy:

~~~sh
bunx wrangler login
bun run deploy:dry-run
bun run deploy
~~~

Both deployment commands rebuild the archive and run the SEO checks before invoking Wrangler. `deploy:dry-run` validates without uploading. `deploy` publishes the `mlp` Worker and prints its workers.dev URL. For a local preview using Cloudflare's routing and headers:

~~~sh
bun run preview:worker
~~~

The deployed Worker is available at [mlp.lucas-chu-barbosa.workers.dev](https://mlp.lucas-chu-barbosa.workers.dev).

The Worker serves Astro's dist/ as [static assets](https://developers.cloudflare.com/workers/framework-guides/web-apps/astro/). All HTML is pre-rendered; no Astro adapter, runtime JavaScript, database, or secrets are required. Chapter URLs redirect to trailing slashes, missing pages return the custom 404 page with a 404 status, and fingerprinted /_astro/ files receive immutable browser caching. Other files use Cloudflare's default revalidation behavior.

After deployment, open the `mlp` Worker in Cloudflare's dashboard, then **Settings → Domains & Routes → Add → Custom Domain**, and enter `mlp.lbxa.net`. This is a separate manual step; deployment does not create that domain or change its DNS. The configuration deliberately omits `routes`, allowing later deployments to preserve dashboard-managed domains. Canonical URLs and the sitemap already use https://mlp.lbxa.net, including when previewing on workers.dev.

## Authoring

- Chapter prose and equations live in src/pages/*.mdx.
- Research.astro provides semantic navigation, a native collapsible table of contents, and shared metadata.
- Seo.astro provides chapter-specific search titles and descriptions, canonical URLs, Open Graph and Twitter cards, and WebSite, Person, WebPage, Article, and BreadcrumbList structured data. Chapters can set seoTitle in frontmatter without changing their visible headings. src/data/site.ts holds the project identity.
- Tailwind 4 supplies spacing and layout utilities. Preflight is deliberately not imported: body type, headings, links, lists, and controls retain browser defaults.
- KaTeX renders math at build time through remark-math and rehype-katex, with accessible MathML and local fonts. Astro 7's unified processor is configured explicitly.
- The direct KaTeX dependency supplies the CSS and must match the version used by rehype-katex. The Astro configuration checks this on startup/build to prevent broken matrix and script positioning after dependency updates.
- Astro's built-in Shiki renders code at build time using the GitHub light high-contrast theme.
- Figure.astro uses astro:assets to emit responsive WebP images with explicit dimensions and lazy loading.
- NetworkDiagram.astro embeds the TikZ network as an accessible, responsive inline SVG. Its caption shares the diagram's maximum width.
- There are no executable client-side scripts, external font requests, analytics, or UI frameworks. JSON-LD is inert structured data.

## Evidence and assets

scripts/prepare-archive.mjs runs before development and production builds. It reads the original notebooks, extracts saved prediction and gradient arrays, computes training metrics, extracts the original cost plot, and copies original diagrams and downloads. Generated copies are ignored by Git.

All in-page bitmap figures are served as WebP. The network diagram and favicon are SVG. Social crawlers receive a separate 1200×630 PNG preview, generated from the TikZ diagram and the directly authored src/assets/social-preview.svg backdrop. `prepare:social` runs before development, checks, and builds; its generated public/social-preview.png is ignored by Git. Original notebook downloads retain their original embedded image data.

The architecture diagram is authored in public/diagrams/feedforward-network.tex using TikZ. To edit it, change that file and run:

~~~sh
bun run prepare:diagram
~~~

Regeneration uses `latex` and `dvisvgm` from TeX Live or MacTeX, with the TikZ, amsmath, and standalone packages. The generated src/assets/diagrams/feedforward-network.svg is checked in alongside the source. Development, checks, and builds verify a fingerprint of the TeX source and compiler script; unchanged diagrams need no TeX installation. Changed diagrams are recompiled, or the build fails with the missing tool requirement. `bun run prepare:diagram --force` forces regeneration after a TeX toolchain update.

The SVG embeds glyph outlines, keeping the diagram sharp at every zoom level without additional fonts or client-side rendering. The original raster drawing remains preserved in the manual; the new captions identify the reconstruction.

The new prediction comparison plot is a checked-in asset generated with Matplotlib from those same archived values. To regenerate it after preparing the archive:

~~~sh
uv run scripts/plot-predictions.py
~~~

The plot script trains no model. The site distinguishes original outputs, derived metrics, and retrospective corrections. The lack of a saved seed, model weights, test predictions, and numerical cost histories is documented in the results chapter.

## Verification

`bun run check:seo` checks the built HTML for unique titles and descriptions, one primary heading per chapter, internal links and anchors, canonical/social URL agreement, structured-data relationships, the social image dimensions, 404 noindex handling, and sitemap coverage. It also confirms that structured data introduces no executable JavaScript. Pass a build directory as an argument to inspect an isolated output.

The 2017 project date describes the original experiment. Add publication/update dates to structured data only when those dates are documented for the web chapters themselves.

With the production preview running at port 4323:

~~~sh
bun run audit
~~~

This runs Lighthouse on every chapter in mobile and desktop configurations. Local Google Chrome must be installed, or CHROME_PATH must identify a compatible binary. AUDIT_URL overrides the preview origin.

HTML and JSON reports plus summary.json are written to reports/lighthouse/. Reports are ignored by Git. Scores describe the local production build; re-run against the deployed site to evaluate hosting and network behavior.

Browser verification should cover chapter links, table-of-contents disclosure and anchors, downloads, narrow-screen overflow, equation rendering, image loading, keyboard access, and absence of client errors.

After deployment, verify the real domain in Google Search Console, submit /sitemap-index.xml, and inspect the public URLs. Indexing and rankings depend on Google's crawl and evaluation, beyond local Lighthouse checks. The metadata follows Google's guidance for [descriptive title links](https://developers.google.com/search/docs/appearance/title-link), [canonical URLs](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls), and [article structured data](https://developers.google.com/search/docs/appearance/structured-data/article).

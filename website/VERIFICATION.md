# Verification

Verified on 27 September 2026 (America/New_York) against the static production build served locally at port 4323, with https://mlp.lbxa.net configured as the production origin. Lighthouse 13.5.0 used its default mobile configuration and its separate desktop configuration.

| Chapter | Mobile performance | Desktop performance | Accessibility | Best practices | SEO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 100 | 100 | 100 | 100 | 100 |
| Problem & foundations | 99 | 100 | 100 | 100 | 100 |
| Data & preparation | 100 | 100 | 100 | 100 | 100 |
| Network architecture | 100 | 100 | 100 | 100 | 100 |
| Learning & backpropagation | 100 | 100 | 100 | 100 | 100 |
| Results & discussion | 100 | 100 | 100 | 100 | 100 |
| Source & reproduction | 100 | 100 | 100 | 100 | 100 |

Accessibility, best-practices, and SEO scores are identical on mobile and desktop. Full HTML and JSON reports and a machine-readable summary are saved locally under reports/lighthouse/; regenerate them with `bun run audit` while the production preview is running.

## Other checks

- Astro check: zero errors, warnings, or hints.
- Production build: seven research chapters, a 404 page, and robots.txt.
- All 181 internal links and referenced assets pass the build-level SEO audit, including anchor targets, original archive downloads, and the TikZ source.
- All seven chapters fit a 320-pixel-wide viewport without document-level horizontal scrolling. Wide tables, equations, and code have their own scrolling regions.
- All displayed bitmap images decode successfully and use Astro-generated WebP URLs, with responsive variants and explicit dimensions. The network reconstruction is an inline SVG compiled from TikZ.
- Browser body font remains Times; no Tailwind reset is loaded.
- All 92 math expressions render without KaTeX errors. Display equations can receive keyboard focus.
- All seven production chapters contain no executable scripts. Each includes one inert JSON-LD block describing the page and its author.
- Native table-of-contents disclosure and in-page navigation were checked in the browser.
- The final build uses https://mlp.lbxa.net for canonical URLs, structured-data identifiers, social image URLs, the seven-page sitemap, and the robots.txt sitemap reference. A separate build verifies the SITE_URL override; its test domain is confined to output/seo-build/.

## Math typesetting review

The KaTeX stylesheet and renderer now both use 0.16.47. A startup/build check rejects mismatched versions, which previously caused overlapping superscripts and matrix entries. Symbolic matrices use an array stretch of 1.5; numerical matrices use 1.25.

All 92 expressions were visually reviewed, and the normalization, forward pass, loss, gradients, finite differences, and error metrics were checked against the original implementation and saved outputs. Notation now consistently distinguishes raw hours and scores from normalized inputs and targets. Transposes are upright and metric sums have explicit limits.

All five chapters containing math were checked at 320, 375, and 1280 pixels. No equation has a parsing error, vertical clipping, or inaccessible left overflow. All display equations fit the 320-pixel viewport after long equations were split into readable lines. Measured matrix rows have positive clearance between their text bounds. Diagnostic equation screenshots are available locally in output/math-review/.

The latest Lighthouse reports cover the pages with these math fixes.

## TikZ architecture reconstruction

The overview and architecture chapters now use a diagram compiled from public/diagrams/feedforward-network.tex. The TikZ source explicitly defines two input nodes, three hidden nodes, one output node, six input-to-hidden connections, and three hidden-to-output connections. Captions identify the reconstruction separately from the original raster drawing.

Both the DVI-to-SVG pipeline and standalone PDF compilation pass. A source-and-script fingerprint triggers regeneration when the diagram changes; the checked-in SVG also passes preparation with TeX tools removed from PATH. The downloadable TeX source matches the authored file byte for byte.

The diagram was checked at 320, 375, and 1280 pixels: the viewBox aspect ratio is preserved, drawing bounds are not clipped, captions match the vector's width and alignment, and neither page overflows horizontally. The SVG has an accessible description, embeds glyph outlines, contains no bitmap, and adds no client scripts or font requests.

Fresh mobile and desktop Lighthouse audits of both affected pages score 100 in performance, accessibility, best practices, and SEO, with zero layout shift and no console errors. These four reports are stored in reports/lighthouse/tikz/.

## Search metadata and sharing

Every chapter has a unique search title (49–61 characters) and description (138–153 characters), a single primary heading, and matching Open Graph and Twitter metadata. A 1200×630 PNG social preview is generated from the TikZ diagram, with its dimensions and accessible description in the metadata. It does not add an in-page image request.

WebSite, Person, WebPage, Article, and chapter BreadcrumbList structured data were parsed and checked for consistent URLs, authorship, headlines, and hierarchy. The 2017 experiment date is not asserted as the publication date of the new web chapters. The 404 is marked noindex and has neither a canonical URL nor article structured data.

`bun run check:seo --require-site` passes against the final build. It verifies canonical and social URLs, JSON-LD relationships, link and anchor targets, the social image, robots.txt, and a sitemap containing exactly the seven canonical chapter URLs. The same checks pass against a separate origin-override build. The publish check was also verified to reject an earlier build without a configured public origin.

The browser confirms that the default Times font, 16-pixel body text, and responsive layout are retained. No executable JavaScript, analytics, or external fonts were added.

## Cloudflare Worker deployment

Deployed on 27 September 2026 with Wrangler 4.141.0 to [mlp.lucas-chu-barbosa.workers.dev](https://mlp.lucas-chu-barbosa.workers.dev). Worker name: `mlp`. Initial version: `fde92d6d-08cf-4363-b3af-3a0b9e76a57c`.

- Astro check and the deployment build pass, along with all seven chapters and 181 internal references in the SEO audit.
- Wrangler's dry run validates the assets-only configuration; the local Worker preview passes routing, metadata, custom 404, and cache checks.
- All seven live chapter responses match the generated HTML byte for byte. All 22 distinct linked routes, assets, and downloads checked return 200.
- All six chapter URLs without trailing slashes redirect to the slash form. Missing URLs return the custom page with HTTP 404 and noindex metadata.
- Fingerprinted /_astro/ assets have immutable caching for one year, and optimized images return the image/webp content type.
- Mobile and desktop Lighthouse runs on the live overview and architecture chapters score **100 in performance, accessibility, best practices, and SEO** in all four runs. Reports are saved under reports/lighthouse/worker/.

The custom domain is intentionally left for manual setup. Canonical URLs, social metadata, and the sitemap use https://mlp.lbxa.net already; no custom domain or DNS record was added during deployment.

## Scope

The seven-chapter table reports local lab measurements. The separate Cloudflare deployment checks above report measurements against the live Worker; neither represents field data. Lighthouse still lists non-scoring opportunities around CSS/font dependency chains and some lazy-loaded images. The research model was not retrained: numerical results come from the original saved notebook outputs. Original files under manual/ and src/ were not changed.

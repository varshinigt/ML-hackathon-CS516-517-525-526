// Screenshot helper for the report: Folium maps (results/maps/*.html) and the evaluator UI.
// Usage: node report/capture.js [--cdn DIR]   (API on :8000 and UI on :8080 must be running)
// --cdn DIR serves Folium's CDN libraries from local npm packages when the machine is offline.
const { chromium } = require('playwright');
const path = require('path'); const fs = require('fs');
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'report', 'img'); fs.mkdirSync(OUT, { recursive: true });
const cdnIdx = process.argv.indexOf('--cdn'); const CDN = cdnIdx > 0 ? process.argv[cdnIdx + 1] : null;
const LOCAL = CDN && {
  'leaflet@1.9.4/dist/leaflet.js': path.join(ROOT, 'ui/vendor/leaflet.js'),
  'leaflet@1.9.4/dist/leaflet.css': path.join(ROOT, 'ui/vendor/leaflet.css'),
  'jquery-3.7.1.min.js': `${CDN}/jquery-3.7.1/package/dist/jquery.min.js`,
  'bootstrap.bundle.min.js': `${CDN}/bootstrap-5.2.2/package/dist/js/bootstrap.bundle.min.js`,
  'bootstrap.min.css': `${CDN}/bootstrap-5.2.2/package/dist/css/bootstrap.min.css`,
  'leaflet.awesome-markers.js': `${CDN}/leaflet.awesome-markers-2.0.5/package/dist/leaflet.awesome-markers.js`,
  'leaflet.awesome-markers.css': `${CDN}/leaflet.awesome-markers-2.0.5/package/dist/leaflet.awesome-markers.css`,
  'all.min.css': `${CDN}/fortawesome-fontawesome-free-6.2.0/package/css/all.min.css`,
};
async function route(page) {
  if (!LOCAL) return;
  await page.route(/^https?:\/\/(cdn\.jsdelivr\.net|code\.jquery\.com|cdnjs\.cloudflare\.com|unpkg\.com)\//, (r) => {
    const url = r.request().url(); const key = Object.keys(LOCAL).find((k) => url.endsWith(k));
    if (key && fs.existsSync(LOCAL[key])) return r.fulfill({ path: LOCAL[key] });
    return r.fulfill({ body: '', contentType: url.endsWith('.css') ? 'text/css' : 'application/javascript' });
  });
  await page.route(/basemaps\.cartocdn\.com|tile\.openstreetmap\.org/, (r) => r.abort());
}
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1300, height: 820 }, deviceScaleFactor: 2 });
  await route(p);
  for (const f of fs.readdirSync(path.join(ROOT, 'results/maps')).filter((f) => f.endsWith('.html'))) {
    await p.goto('file://' + path.join(ROOT, 'results/maps', f)); await p.waitForTimeout(2500);
    await p.screenshot({ path: path.join(OUT, f.replace('.html', '.png')) }); console.log('map', f);
  }
  await p.goto('http://localhost:8080'); await p.click('#connect'); await p.waitForSelector('.trip');
  await p.click('.trip >> nth=0'); await p.waitForSelector('.big', { timeout: 120000 }); await p.waitForTimeout(1500);
  await p.screenshot({ path: path.join(OUT, 'ui.png') }); console.log('ui');
  await b.close();
})();

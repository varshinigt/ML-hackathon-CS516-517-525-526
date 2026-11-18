// Render report/report.html to report/report.pdf with headless Chromium (Playwright).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + path.join(__dirname, 'report.html'), { waitUntil: 'load' });
  await p.pdf({ path: path.join(__dirname, 'report.pdf'), format: 'A4', printBackground: true,
                margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' },
                displayHeaderFooter: true, headerTemplate: '<span></span>',
                footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888">PathSnap technical report · page <span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  await b.close(); console.log('wrote report/report.pdf');
})();

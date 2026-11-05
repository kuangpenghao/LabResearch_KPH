const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const puppeteer = require('./runtime/node_modules/puppeteer-core');
const root = __dirname;
const moduleRoot = path.join(root, 'runtime', 'node_modules');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
const server = http.createServer((req, res) => {
  if (req.url === '/') { res.setHeader('Content-Type','text/html'); res.end('<html><body></body></html>'); return; }
  const resource = path.resolve(moduleRoot, '.' + decodeURIComponent(req.url.split('?')[0]));
  if (!resource.startsWith(moduleRoot + path.sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader('Content-Type', resource.endsWith('.js') || resource.endsWith('.mjs') ? 'text/javascript' : 'text/html');
    res.end(fs.readFileSync(resource));
  } catch { res.writeHead(404).end(); }
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await puppeteer.launch({executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', headless:true,
      userDataDir:path.join(root,'browser_profile'),args:['--no-first-run','--disable-extensions','--disable-background-networking']});
    const page = await browser.newPage();
    await page.setViewport({width:1800,height:1400,deviceScaleFactor:1.3});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.setContent('<html><head><meta charset="UTF-8"></head><body style="margin:0;padding:24px;background:white;font-family:Microsoft YaHei,Segoe UI,sans-serif"><div id="chart" style="display:inline-block"></div></body></html>');
    await page.evaluate(async url => {
      window.mermaid = (await import(url)).default;
      window.mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',fontFamily:'Microsoft YaHei, Segoe UI, sans-serif',
        themeVariables:{primaryColor:'#eef5ff',primaryTextColor:'#173151',primaryBorderColor:'#6b89a9',lineColor:'#53687d',secondaryColor:'#edf7f1',tertiaryColor:'#f6f8fb',fontSize:'16px'},
        flowchart:{useMaxWidth:false,htmlLabels:true,nodeSpacing:28,rankSpacing:36,curve:'basis'}});
    }, `http://127.0.0.1:${server.address().port}/mermaid/dist/mermaid.esm.min.mjs`);
    const results=[];
    for(let i=0;i<manifest.length;i++) {
      const item=manifest[i];
      const result=await page.evaluate(async ({code,id})=>{
        await window.mermaid.parse(code);
        const {svg}=await window.mermaid.render(id,code);
        document.getElementById('chart').innerHTML=svg;
        const el=document.querySelector('#chart svg');
        const box=el.viewBox.baseVal;
        el.style.width=box.width+'px';el.style.height=box.height+'px';el.style.maxWidth='none';
        return {svg:el.outerHTML,width:box.width,height:box.height};
      },{code:item.code,id:'diagram'+i});
      const stem=path.join(item.directory,'diagram_'+item.number);
      fs.writeFileSync(stem+'.svg',result.svg);
      const chart=await page.$('#chart');
      await chart.screenshot({path:stem+'.png'});
      results.push({paper:item.paper,number:item.number,syntax_valid:true,width:result.width,height:result.height,png:stem+'.png'});
      console.log(item.paper,item.number,Math.round(result.width)+'x'+Math.round(result.height));
    }
    fs.writeFileSync(path.join(root,'render_verification.json'),JSON.stringify(results,null,2));
  } finally { if(browser) await browser.close(); server.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});

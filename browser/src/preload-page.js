// Runs in every web page (sandboxed, isolated world). Reports only:
//   - how far the user scrolled (max ratio), throttled
//   - text the user copied
//   - that the user is active (to pause dwell when idle)
// It never reads form fields, keystrokes or cookies.
'use strict';
const { ipcRenderer } = require('electron');

let lastScroll = 0;
let pending = false;
function reportScroll() {
  pending = false;
  const doc = document.documentElement;
  const max = Math.max(1, doc.scrollHeight - window.innerHeight);
  const ratio = Math.min(1, (window.scrollY + 0) / max);
  if (ratio > lastScroll + 0.02 || (doc.scrollHeight <= window.innerHeight && lastScroll === 0)) {
    lastScroll = doc.scrollHeight <= window.innerHeight ? 1 : ratio;
    ipcRenderer.send('page:scroll', lastScroll);
  }
}
window.addEventListener('scroll', () => { if (!pending) { pending = true; setTimeout(reportScroll, 500); } }, { passive: true });
window.addEventListener('load', () => setTimeout(reportScroll, 1000));

let lastActivity = 0;
const active = () => {
  const now = Date.now();
  if (now - lastActivity > 5000) { lastActivity = now; ipcRenderer.send('page:activity'); }
};
for (const ev of ['mousemove', 'wheel', 'keydown', 'touchstart']) window.addEventListener(ev, active, { passive: true, capture: true });

document.addEventListener('copy', (e) => {
  const t = e.target;
  if (t && (t.type === 'password' || (t.closest && t.closest('input[type=password]')))) return;
  ipcRenderer.send('page:copy', String(window.getSelection()));
}, true);

ipcRenderer.on('page:flash', (_e, msg) => {
  const el = document.createElement('div');
  el.textContent = '✦ ' + msg;
  el.style.cssText = 'position:fixed;z-index:2147483647;right:16px;bottom:16px;padding:8px 12px;border-radius:8px;' +
    'background:#1f2330;color:#fff;font:13px system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.25);transition:opacity .4s';
  document.documentElement.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 500); }, 1200);
});

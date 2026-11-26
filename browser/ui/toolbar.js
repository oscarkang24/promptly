'use strict';
const P = window.promptly;
const $ = (id) => document.getElementById(id);
const isPrivate = new URLSearchParams(location.search).get('private') === '1';
if (isPrivate) { document.body.classList.add('private'); $('priv').hidden = false; $('mem').hidden = true; }

let state = { tabs: [] };
let editing = false;

function render() {
  const wrap = $('tabs');
  wrap.replaceChildren();
  for (const t of state.tabs) {
    const el = document.createElement('div');
    el.className = 'tab' + (t.active ? ' on' : '') + (t.owner === 'agent' ? ' agent' : '');
    el.title = t.url;
    if (t.favicon) { const img = document.createElement('img'); img.src = t.favicon; img.onerror = () => img.remove(); el.append(img); }
    const title = document.createElement('span');
    title.className = 't';
    title.textContent = t.loading ? 'Loading…' : (t.title || t.url || 'New tab');
    el.append(title);
    if (t.owner === 'agent' || t.agentMayDrive) {
      const b = document.createElement('span'); b.className = 'badge'; b.textContent = 'agent'; el.append(b);
    }
    const x = document.createElement('span');
    x.className = 'x'; x.textContent = '×';
    x.onclick = (e) => { e.stopPropagation(); P.tab_close(t.id); };
    el.append(x);
    el.onclick = () => P.tab_activate(t.id);
    el.onauxclick = (e) => { if (e.button === 1) P.tab_close(t.id); };
    wrap.append(el);
  }
  const nb = document.createElement('button');
  nb.id = 'new'; nb.textContent = '+'; nb.title = 'New tab (Ctrl+T)'; nb.onclick = () => P.tab_new();
  wrap.append(nb);

  const a = state.tabs.find((t) => t.active);
  $('back').disabled = !a || !a.canGoBack;
  $('fwd').disabled = !a || !a.canGoForward;
  if (!editing) $('addr').value = a && a.url !== 'about:blank' ? a.url : '';
  $('mem').classList.toggle('on', !!state.panelOpen);
  const drive = $('drive');
  drive.hidden = isPrivate || !a || a.owner === 'agent' || a.url === 'about:blank';
  if (a) drive.textContent = a.agentMayDrive ? '🤖 Agent may drive · stop' : '🤖 Let agent drive';
}

P.on('state', (s) => { state = s; render(); });
P.on('focus-address', () => { $('addr').focus(); $('addr').select(); });
$('addr').addEventListener('focus', () => { editing = true; $('addr').select(); });
$('addr').addEventListener('blur', () => { editing = false; render(); });
$('addr').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { P.tab_navigate($('addr').value); $('addr').blur(); }
  if (e.key === 'Escape') { $('addr').blur(); }
});
$('back').onclick = () => P.tab_back();
$('fwd').onclick = () => P.tab_forward();
$('reload').onclick = () => P.tab_reload();
$('mem').onclick = () => P.panel_toggle();
$('drive').onclick = () => { const a = state.tabs.find((t) => t.active); if (a) P.tab_agentDrive(a.id, !a.agentMayDrive); };
P.state_get();

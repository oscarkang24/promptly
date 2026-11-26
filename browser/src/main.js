// Promptly Browser: main process.
//
//  - Tabs are WebContentsViews under one toolbar view and an optional memory side panel.
//  - Each tab reports visits, visible dwell time, scroll depth, readable text, searches,
//    copies and explicit highlights to the local Promptly daemon (127.0.0.1:8787).
//  - A control server on 127.0.0.1 lets agents list/read tabs and drive the tabs they own.
//  - Private windows (Ctrl/Cmd+Shift+N) use an in-memory session and record nothing.
'use strict';

const { app, BaseWindow, WebContentsView, ipcMain, Menu, clipboard, powerMonitor, session, shell } = require('electron');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const lib = require('./lib');

const HOME = process.env.PROMPTLY_HOME || path.join(os.homedir(), '.promptly');
const DAEMON = process.env.PROMPTLY_DAEMON || 'http://127.0.0.1:8787';
const TOOLBAR_H = 84;
const PANEL_W = 380;
const IDLE_AFTER_S = 60;   // no input for this long: dwell stops counting
const TICK_MS = 2000;
const HOME_PAGE = 'about:blank';

if (process.env.PROMPTLY_CDP_PORT) {
  // For Playwright/Puppeteer agents: chromium.connectOverCDP('http://127.0.0.1:<port>')
  app.commandLine.appendSwitch('remote-debugging-port', process.env.PROMPTLY_CDP_PORT);
  app.commandLine.appendSwitch('remote-debugging-address', '127.0.0.1');
}

// ---------------------------------------------------------------- daemon client
let TOKEN = null;
let daemonProc = null;

function readToken() {
  try { return fs.readFileSync(path.join(HOME, 'browser.token'), 'utf8').trim() || null; } catch { return null; }
}

async function daemon(method, p, body) {
  if (!TOKEN) TOKEN = readToken();
  if (!TOKEN) return null;
  try {
    const r = await fetch(DAEMON + p, {
      method,
      headers: { 'X-Promptly-Token': TOKEN, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (r.status === 401) { TOKEN = readToken(); return null; }
    const ct = r.headers.get('content-type') || '';
    return ct.includes('json') ? r.json() : r.text();
  } catch {
    return null; // daemon down: browsing keeps working, nothing is recorded
  }
}

async function ensureDaemon() {
  try {
    const r = await fetch(DAEMON + '/api/status');
    if (r.ok) { TOKEN = readToken(); return true; }
  } catch { /* not running */ }
  const cmd = process.env.PROMPTLY_CMD || 'promptly';
  const port = new URL(DAEMON).port || '8787';
  daemonProc = spawn(cmd, ['serve', '--port', port], { stdio: 'ignore', env: { ...process.env, PROMPTLY_HOME: HOME } });
  daemonProc.on('error', () => { daemonProc = null; console.warn(`[promptly] could not start "${cmd} serve"; memory is off`); });
  for (let i = 0; i < 40; i++) {
    await new Promise((res) => setTimeout(res, 250));
    try {
      const r = await fetch(DAEMON + '/api/status');
      if (r.ok) { TOKEN = readToken(); return true; }
    } catch { /* keep waiting */ }
  }
  return false;
}

// ---------------------------------------------------------------- windows & tabs
/** @type {Map<number, Win>} */
const windows = new Map();
/** @type {Map<string, Tab>} */
const tabs = new Map();
let lastInputAt = Date.now();

class Win {
  constructor({ private: isPrivate = false } = {}) {
    this.private = isPrivate;
    this.panelOpen = false;
    this.tabIds = [];
    this.activeId = null;
    this.win = new BaseWindow({ width: 1320, height: 860, title: isPrivate ? 'Promptly (private)' : 'Promptly' });
    const uiPrefs = { preload: path.join(__dirname, 'preload-ui.js'), contextIsolation: true, sandbox: true };
    this.toolbar = new WebContentsView({ webPreferences: uiPrefs });
    this.panel = new WebContentsView({ webPreferences: uiPrefs });
    this.win.contentView.addChildView(this.toolbar);
    this.toolbar.webContents.loadFile(path.join(__dirname, '..', 'ui', 'toolbar.html'), { query: { private: isPrivate ? '1' : '0' } });
    this.panel.webContents.loadFile(path.join(__dirname, '..', 'ui', 'panel.html'));
    this.session = isPrivate ? session.fromPartition('promptly-private-' + crypto.randomUUID()) : session.defaultSession;
    this.win.on('resize', () => this.layout());
    this.win.on('focus', () => { lastInputAt = Date.now(); });
    this.win.on('closed', () => {
      for (const id of [...this.tabIds]) closeTab(id, { silent: true });
      windows.delete(this.win.id);
    });
    windows.set(this.win.id, this);
  }

  layout() {
    const { width, height } = this.win.getContentBounds();
    const pw = this.panelOpen ? PANEL_W : 0;
    this.toolbar.setBounds({ x: 0, y: 0, width, height: TOOLBAR_H });
    this.panel.setBounds({ x: width - pw, y: TOOLBAR_H, width: pw, height: height - TOOLBAR_H });
    for (const id of this.tabIds) {
      const t = tabs.get(id);
      t.view.setBounds({ x: 0, y: TOOLBAR_H, width: width - pw, height: height - TOOLBAR_H });
      t.view.setVisible(id === this.activeId);
    }
  }

  togglePanel(open = !this.panelOpen) {
    this.panelOpen = open;
    if (open) this.win.contentView.addChildView(this.panel);
    else this.win.contentView.removeChildView(this.panel);
    this.layout();
    if (open) this.panel.webContents.send('panel:refresh');
  }

  pushState() {
    const state = {
      private: this.private,
      panelOpen: this.panelOpen,
      activeId: this.activeId,
      tabs: this.tabIds.map((id) => tabs.get(id).summary()),
    };
    this.toolbar.webContents.send('state', state);
    if (this.panelOpen) this.panel.webContents.send('state', state);
  }
}

class Tab {
  constructor(win, { actor = 'user', url = HOME_PAGE } = {}) {
    this.id = crypto.randomUUID().slice(0, 8);
    this.win = win;
    this.owner = actor;           // who opened it
    this.agentMayDrive = actor === 'agent';
    this.nextActor = actor;       // who caused the next navigation
    this.visitId = null;
    this.visitDwell = 0;
    this.visitScroll = 0;
    this.prevUrl = null;
    this.contentTimer = null;
    this.view = new WebContentsView({
      webPreferences: {
        session: win.session,
        preload: path.join(__dirname, 'preload-page.js'),
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
      },
    });
    const wc = this.view.webContents;
    wc.setWindowOpenHandler(({ url: u }) => {
      openTab(this.win, u, { actor: this.owner === 'agent' ? 'agent' : 'user', background: false });
      return { action: 'deny' };
    });
    wc.on('did-navigate', (_e, u) => this.onNavigate(u, false));
    wc.on('did-navigate-in-page', (_e, u, isMain) => { if (isMain) this.onNavigate(u, true); });
    wc.on('page-title-updated', () => { this.win.pushState(); this.scheduleContent(1500); });
    wc.on('did-finish-load', () => this.scheduleContent(2000));
    wc.on('did-start-loading', () => this.win.pushState());
    wc.on('did-stop-loading', () => this.win.pushState());
    wc.on('page-favicon-updated', (_e, icons) => { this.favicon = icons[0]; this.win.pushState(); });
    wc.on('before-input-event', (_e, input) => {
      lastInputAt = Date.now();
      if (input.type === 'keyDown') this.nextActor = 'user';
      if (input.type === 'keyDown' && input.shift && (input.control || input.meta) && input.key.toLowerCase() === 'h') rememberSelection(this);
    });
    wc.on('input-event', (_e, input) => {
      if (input.type === 'mouseDown') { this.nextActor = 'user'; lastInputAt = Date.now(); }
    });
    wc.on('context-menu', (_e, params) => this.contextMenu(params));
    win.win.contentView.addChildView(this.view);
    tabs.set(this.id, this);
    if (url) wc.loadURL(url);
  }

  get url() { return this.view.webContents.getURL(); }
  get title() { return this.view.webContents.getTitle(); }

  summary() {
    const wc = this.view.webContents;
    return {
      id: this.id, url: this.url, title: this.title, loading: wc.isLoading(),
      canGoBack: wc.navigationHistory.canGoBack(), canGoForward: wc.navigationHistory.canGoForward(),
      owner: this.owner, agentMayDrive: this.agentMayDrive, favicon: this.favicon || null,
      active: this.win.activeId === this.id,
    };
  }

  async onNavigate(url, inPage) {
    if (inPage && this.prevUrl && stripHash(this.prevUrl) === stripHash(url)) return;
    await this.endVisit();
    const actor = this.nextActor;
    this.nextActor = this.owner === 'agent' && !this.userTookOver ? 'agent' : 'user';
    const from = this.prevUrl;
    this.prevUrl = url;
    this.visitDwell = 0;
    this.visitScroll = 0;
    this.win.pushState();
    if (this.win.private || !lib.isRecordable(url)) return;
    const r = await daemon('POST', '/api/browser/visit/start', { url, title: this.title, tab_id: this.id, actor, from_url: from });
    this.visitId = r && r.visit_id;
    const q = lib.searchQuery(url);
    if (q) daemon('POST', '/api/browser/event', { kind: 'search', text: q, url, actor });
  }

  async endVisit() {
    if (this.visitId) {
      const id = this.visitId;
      this.visitId = null;
      await daemon('POST', '/api/browser/visit/end', { visit_id: id, dwell_ms: Math.round(this.visitDwell), scroll: this.visitScroll });
    }
  }

  scheduleContent(ms) {
    clearTimeout(this.contentTimer);
    if (this.win.private || !lib.isRecordable(this.url)) return;
    this.contentTimer = setTimeout(async () => {
      try {
        const page = await this.view.webContents.executeJavaScript(`(${lib.extractReadable.toString()})(60000)`);
        if (page && lib.isRecordable(page.url)) daemon('POST', '/api/browser/content', { url: page.url, title: page.title, text: page.text });
      } catch { /* page navigated away or blocked script */ }
    }, ms);
  }

  contextMenu(params) {
    const items = [];
    if (params.selectionText && params.selectionText.trim()) {
      items.push({ label: 'Remember this highlight  (Ctrl+Shift+H)', click: () => rememberText(this, params.selectionText) });
      items.push({ label: 'Copy', role: 'copy' });
      items.push({ type: 'separator' });
    }
    if (params.linkURL) items.push({ label: 'Open link in new tab', click: () => openTab(this.win, params.linkURL, { background: true }) });
    if (params.isEditable) items.push({ label: 'Paste', role: 'paste' });
    items.push({ label: 'Back', enabled: this.view.webContents.navigationHistory.canGoBack(), click: () => this.view.webContents.navigationHistory.goBack() });
    items.push({ label: 'Reload', click: () => this.view.webContents.reload() });
    items.push({ label: 'Inspect', click: () => this.view.webContents.inspectElement(params.x, params.y) });
    Menu.buildFromTemplate(items).popup();
  }
}

function stripHash(u) { return (u || '').split('#')[0]; }

function openTab(win, url, { actor = 'user', background = false } = {}) {
  const t = new Tab(win, { actor, url: url || HOME_PAGE });
  const idx = win.activeId ? win.tabIds.indexOf(win.activeId) + 1 : win.tabIds.length;
  win.tabIds.splice(idx, 0, t.id);
  if (!background || !win.activeId) activateTab(t.id);
  else win.layout();
  win.pushState();
  return t;
}

function activateTab(id) {
  const t = tabs.get(id);
  if (!t) return;
  t.win.activeId = id;
  t.win.layout();
  t.win.pushState();
  if (t.url === HOME_PAGE) t.win.toolbar.webContents.send('focus-address');
  else t.view.webContents.focus();
}

async function closeTab(id, { silent = false } = {}) {
  const t = tabs.get(id);
  if (!t) return;
  const win = t.win;
  clearTimeout(t.contentTimer);
  await t.endVisit();
  tabs.delete(id);
  const i = win.tabIds.indexOf(id);
  win.tabIds.splice(i, 1);
  if (!silent) {
    try { win.win.contentView.removeChildView(t.view); } catch { /* window closing */ }
    t.view.webContents.close();
    if (!win.tabIds.length) return win.win.close();
    if (win.activeId === id) activateTab(win.tabIds[Math.min(i, win.tabIds.length - 1)]);
    win.pushState();
  }
}

async function rememberSelection(tab) {
  const text = await tab.view.webContents.executeJavaScript('String(window.getSelection())').catch(() => '');
  rememberText(tab, text);
}

function rememberText(tab, text) {
  text = (text || '').trim();
  if (!text || tab.win.private) return;
  daemon('POST', '/api/browser/event', { kind: 'highlight', text, url: tab.url, actor: 'user' });
  tab.view.webContents.send('page:flash', 'Remembered');
}

// Count visible, attended time on the active tab of the focused window.
setInterval(() => {
  const idle = (powerMonitor.getSystemIdleTime ? powerMonitor.getSystemIdleTime() : 0) >= IDLE_AFTER_S
    || Date.now() - lastInputAt > IDLE_AFTER_S * 1000 * 5;
  for (const w of windows.values()) {
    if (!w.win.isFocused() || w.win.isMinimized() || idle) continue;
    const t = tabs.get(w.activeId);
    if (t && t.visitId) t.visitDwell += TICK_MS;
  }
}, TICK_MS);

// ---------------------------------------------------------------- IPC from pages and UI
function tabForSender(sender) {
  for (const t of tabs.values()) if (t.view.webContents === sender) return t;
  return null;
}
function winForSender(sender) {
  for (const w of windows.values()) if (w.toolbar.webContents === sender || w.panel.webContents === sender) return w;
  return null;
}

ipcMain.on('page:scroll', (e, ratio) => {
  const t = tabForSender(e.sender);
  if (t && typeof ratio === 'number') t.visitScroll = Math.max(t.visitScroll, Math.min(1, Math.max(0, ratio)));
});
ipcMain.on('page:copy', (e, text) => {
  const t = tabForSender(e.sender);
  if (t && !t.win.private && typeof text === 'string' && text.trim().length >= 3 && lib.isRecordable(t.url))
    daemon('POST', '/api/browser/event', { kind: 'copy', text: text.slice(0, 2000), url: t.url, actor: 'user' });
});
ipcMain.on('page:activity', (e) => { if (tabForSender(e.sender)) lastInputAt = Date.now(); });

const uiHandlers = {
  'tab:new': (w, url) => openTab(w, url ? lib.resolveInput(url) : HOME_PAGE),
  'tab:close': (w, id) => closeTab(id),
  'tab:activate': (w, id) => activateTab(id),
  'tab:navigate': (w, input) => {
    const t = tabs.get(w.activeId);
    const url = lib.resolveInput(input);
    if (t && url) { t.nextActor = 'user'; t.userTookOver = true; t.view.webContents.loadURL(url); t.view.webContents.focus(); }
  },
  'tab:back': (w) => tabs.get(w.activeId)?.view.webContents.navigationHistory.goBack(),
  'tab:forward': (w) => tabs.get(w.activeId)?.view.webContents.navigationHistory.goForward(),
  'tab:reload': (w) => tabs.get(w.activeId)?.view.webContents.reload(),
  'tab:agentDrive': (w, id, allow) => { const t = tabs.get(id); if (t && t.owner === 'user') { t.agentMayDrive = !!allow; w.pushState(); } },
  'panel:toggle': (w) => w.togglePanel(),
  'window:private': () => openWindow({ private: true }),
  'state:get': (w) => w.pushState(),
  // Panel → daemon. The renderer never sees the token.
  'mem:stats': () => daemon('GET', '/api/browser/stats'),
  'mem:policy': () => daemon('GET', '/api/browser/policy'),
  'mem:pause': (w, paused) => daemon('POST', '/api/browser/policy', { paused: !!paused }),
  'mem:block': async (w) => {
    const t = tabs.get(w.activeId);
    if (!t || !lib.isRecordable(t.url)) return null;
    const host = new URL(t.url).hostname.replace(/^www\./, '');
    const pol = await daemon('GET', '/api/browser/policy');
    if (!pol) return null;
    await daemon('POST', '/api/browser/forget', { domain: host });
    return daemon('POST', '/api/browser/policy', { blocked: [...new Set([...pol.blocked, host])] });
  },
  'mem:forgetSite': (w) => {
    const t = tabs.get(w.activeId);
    if (!t || !lib.isRecordable(t.url)) return null;
    return daemon('POST', '/api/browser/forget', { domain: new URL(t.url).hostname });
  },
  'mem:forgetRecent': (w, minutes) => daemon('POST', '/api/browser/forget', { since_minutes: Number(minutes) || 60 }),
  'mem:recent': () => daemon('GET', '/api/browser/recent?limit=15'),
  'mem:interests': () => daemon('GET', '/api/browser/interests'),
  'mem:search': (w, q) => daemon('GET', '/api/browser/search?q=' + encodeURIComponent(String(q || '')) + '&k=10'),
  'mem:context': (w, task) => daemon('POST', '/api/browser/context', { task: String(task || ''), format: 'markdown' }),
  'mem:suggest': () => daemon('GET', '/api/browser/suggest-facts'),
  'mem:accept': (w, fact) => daemon('POST', '/api/learn/accept', { facts: [fact] }),
  'clip:write': (w, text) => clipboard.writeText(String(text || '')),
  'open:external': (w, url) => { if (lib.isRecordable(url)) openTab(w, url); },
};

for (const [name, fn] of Object.entries(uiHandlers)) {
  ipcMain.handle(name, (e, ...args) => {
    const w = winForSender(e.sender);
    if (!w) throw new Error('not a Promptly UI view');
    if (w.private && name.startsWith('mem:')) return null;
    return fn(w, ...args);
  });
}

// ---------------------------------------------------------------- agent control server
function focusedWin() {
  const f = BaseWindow.getFocusedWindow();
  const w = (f && windows.get(f.id)) || [...windows.values()].find((x) => !x.private);
  return w && !w.private ? w : null;
}

function mustDrive(id) {
  const t = tabs.get(id);
  if (!t || t.win.private) throw httpError(404, 'no such tab');
  if (!t.agentMayDrive) throw httpError(403, 'this is the user\'s tab; open your own with POST /tabs, or ask the user to choose "Let agent drive"');
  return t;
}

function httpError(code, msg) { const e = new Error(msg); e.code = code; return e; }

const routes = [
  { method: 'GET', path: '/tabs', handler: () => [...tabs.values()].filter((t) => !t.win.private).map((t) => t.summary()) },
  {
    method: 'POST', path: '/tabs', handler: (_p, body) => {
      const w = focusedWin() || openWindow();
      const url = lib.resolveInput(body.url);
      if (!url || !lib.isRecordable(url)) throw httpError(400, 'url must be http(s)');
      const t = openTab(w, url, { actor: 'agent', background: body.background !== false });
      return { id: t.id, url };
    },
  },
  {
    method: 'GET', path: '/tabs/:id/read', handler: async (p, _b, query) => {
      const t = tabs.get(p.id);
      if (!t || t.win.private) throw httpError(404, 'no such tab');
      const max = Math.min(200000, Math.max(500, Number(query.get('max_chars')) || 20000));
      return t.view.webContents.executeJavaScript(`(${lib.extractReadable.toString()})(${max})`);
    },
  },
  {
    method: 'POST', path: '/tabs/:id/navigate', handler: async (p, body) => {
      const t = mustDrive(p.id);
      const url = lib.resolveInput(body.url);
      if (!url || !lib.isRecordable(url)) throw httpError(400, 'url must be http(s)');
      t.nextActor = 'agent';
      await t.view.webContents.loadURL(url).catch(() => {});
      return { ok: true, url: t.url, title: t.title };
    },
  },
  {
    method: 'POST', path: '/tabs/:id/click', handler: async (p, body) => {
      const t = mustDrive(p.id);
      t.nextActor = 'agent';
      return t.view.webContents.executeJavaScript(lib.clickScript(body.selector));
    },
  },
  {
    method: 'POST', path: '/tabs/:id/type', handler: async (p, body) => {
      const t = mustDrive(p.id);
      t.nextActor = 'agent';
      return t.view.webContents.executeJavaScript(lib.typeScript(body.selector, body.text, !!body.submit));
    },
  },
  {
    method: 'GET', path: '/tabs/:id/screenshot', handler: async (p) => {
      const t = tabs.get(p.id);
      if (!t || t.win.private) throw httpError(404, 'no such tab');
      const img = await t.view.webContents.capturePage();
      return { png_base64: img.toPNG().toString('base64') };
    },
  },
  {
    method: 'DELETE', path: '/tabs/:id', handler: async (p) => {
      const t = tabs.get(p.id);
      if (!t || t.win.private) throw httpError(404, 'no such tab');
      if (t.owner !== 'agent') throw httpError(403, 'agents can only close tabs they opened');
      await closeTab(p.id);
      return { ok: true };
    },
  },
];

function startControlServer() {
  const srv = http.createServer(async (req, res) => {
    const send = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };
    // Web pages may try to reach localhost: refuse anything a browser would label with an Origin.
    if (req.headers.origin) return send(403, { error: 'cross-origin requests are not allowed' });
    if (!lib.safeEqual(req.headers['x-promptly-token'], TOKEN || readToken() || '')) return send(401, { error: 'bad token' });
    const u = new URL(req.url, 'http://127.0.0.1');
    const m = lib.matchRoute(routes, req.method, u.pathname);
    if (!m) return send(404, { error: 'not found' });
    let raw = '';
    for await (const chunk of req) { raw += chunk; if (raw.length > 1e6) return send(413, { error: 'too large' }); }
    try {
      const body = raw ? JSON.parse(raw) : {};
      send(200, await m.handler(m.params, body, u.searchParams));
    } catch (e) {
      send(e.code && Number.isInteger(e.code) ? e.code : 500, { error: String(e.message || e) });
    }
  });
  srv.listen(Number(process.env.PROMPTLY_CONTROL_PORT) || 0, '127.0.0.1', () => {
    const { port } = srv.address();
    fs.mkdirSync(HOME, { recursive: true });
    fs.writeFileSync(path.join(HOME, 'shell.json'), JSON.stringify({ port, pid: process.pid }), { mode: 0o600 });
    console.log(`[promptly] agent control on 127.0.0.1:${port}`);
  });
  return srv;
}

// ---------------------------------------------------------------- app lifecycle
function openWindow(opts = {}) {
  const w = new Win(opts);
  w.layout();
  openTab(w, process.env.PROMPTLY_START_URL || HOME_PAGE);
  return w;
}

function appMenu() {
  const cur = () => focusedWin() || [...windows.values()][0];
  const curAny = () => { const f = BaseWindow.getFocusedWindow(); return (f && windows.get(f.id)) || [...windows.values()][0]; };
  return Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    {
      label: 'File', submenu: [
        { label: 'New Tab', accelerator: 'CmdOrCtrl+T', click: () => { const w = curAny() || openWindow(); openTab(w, HOME_PAGE); } },
        { label: 'New Window', accelerator: 'CmdOrCtrl+N', click: () => openWindow() },
        { label: 'New Private Window', accelerator: 'CmdOrCtrl+Shift+N', click: () => openWindow({ private: true }) },
        { label: 'Close Tab', accelerator: 'CmdOrCtrl+W', click: () => { const w = curAny(); if (w) closeTab(w.activeId); } },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    {
      label: 'View', submenu: [
        { label: 'Focus Address Bar', accelerator: 'CmdOrCtrl+L', click: () => curAny()?.toolbar.webContents.send('focus-address') },
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => { const w = curAny(); tabs.get(w?.activeId)?.view.webContents.reload(); } },
        { label: 'Memory Panel', accelerator: 'CmdOrCtrl+Shift+M', click: () => curAny()?.togglePanel() },
        { label: 'Next Tab', accelerator: 'Ctrl+Tab', click: () => { const w = curAny(); if (!w) return; const i = w.tabIds.indexOf(w.activeId); activateTab(w.tabIds[(i + 1) % w.tabIds.length]); } },
        { type: 'separator' },
        { role: 'toggleDevTools' },
      ],
    },
    {
      label: 'Memory', submenu: [
        { label: 'Remember Highlight', accelerator: 'CmdOrCtrl+Shift+H', click: () => { const w = curAny(); const t = tabs.get(w?.activeId); if (t) rememberSelection(t); } },
        { label: 'Pause Recording', type: 'checkbox', click: (item) => daemon('POST', '/api/browser/policy', { paused: item.checked }) },
        { label: 'Forget Last Hour', click: () => daemon('POST', '/api/browser/forget', { since_minutes: 60 }) },
      ],
    },
  ]);
}

app.whenReady().then(async () => {
  Menu.setApplicationMenu(appMenu());
  ensureDaemon().then((ok) => { if (!ok) console.warn('[promptly] memory daemon unavailable; browsing without memory'); });
  TOKEN = readToken();
  // The daemon creates the token on first start; until then the control server uses a fresh one.
  if (!TOKEN) {
    TOKEN = crypto.randomBytes(24).toString('base64url');
    fs.mkdirSync(HOME, { recursive: true });
    try { fs.writeFileSync(path.join(HOME, 'browser.token'), TOKEN, { mode: 0o600, flag: 'wx' }); } catch { TOKEN = readToken(); }
  }
  startControlServer();
  openWindow();
});

app.on('web-contents-created', (_e, wc) => {
  wc.on('will-attach-webview', (ev) => ev.preventDefault());
  // UI views must never navigate away from their bundled pages.
  wc.on('will-navigate', (ev, url) => {
    if (url.startsWith('file:') && !tabForSender(wc)) return;
    if (!tabForSender(wc)) { ev.preventDefault(); if (lib.isRecordable(url)) shell.openExternal(url); }
  });
});

let quitting = false;
app.on('before-quit', (e) => {
  if (quitting) return;
  // Flush open visits (dwell time) before exiting.
  e.preventDefault();
  quitting = true;
  const flush = Promise.all([...tabs.values()].map((t) => t.endVisit()));
  Promise.race([flush, new Promise((r) => setTimeout(r, 1500))]).finally(() => {
    try { fs.unlinkSync(path.join(HOME, 'shell.json')); } catch { /* already gone */ }
    if (daemonProc) daemonProc.kill();
    app.quit();
  });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (!windows.size) openWindow(); });

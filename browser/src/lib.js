// Pure helpers shared by the main process. No Electron imports, so `node --test` can cover them.
'use strict';

const SEARCH_ENGINES = [
  { host: /(^|\.)google\.[a-z.]+$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)bing\.com$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)duckduckgo\.com$/, path: /^\/$|^\/html/, param: 'q' },
  { host: /(^|\.)search\.brave\.com$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)kagi\.com$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)github\.com$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)stackoverflow\.com$/, path: /^\/search/, param: 'q' },
  { host: /(^|\.)youtube\.com$/, path: /^\/results/, param: 'search_query' },
  { host: /(^|\.)wikipedia\.org$/, path: /^\/w\/index\.php/, param: 'search' },
];

/** If `url` is a search results page, return the query the user searched for. */
function searchQuery(url) {
  let u;
  try { u = new URL(url); } catch { return null; }
  for (const e of SEARCH_ENGINES) {
    if (e.host.test(u.hostname) && e.path.test(u.pathname)) {
      const q = (u.searchParams.get(e.param) || '').trim();
      return q || null;
    }
  }
  return null;
}

const DEFAULT_SEARCH = 'https://duckduckgo.com/?q=';

/** Turn what the user typed in the address bar into a URL. */
function resolveInput(text, searchPrefix = DEFAULT_SEARCH) {
  const t = (text || '').trim();
  if (!t) return null;
  if (/^(https?|file|about):/i.test(t)) return t;
  if (/^localhost(:\d+)?(\/|$)/i.test(t) || /^\d{1,3}(\.\d{1,3}){3}(:\d+)?(\/|$)/.test(t)) return 'http://' + t;
  if (!/\s/.test(t) && /^[^/]+\.[a-z]{2,}(:\d+)?(\/.*)?$/i.test(t)) return 'https://' + t;
  return searchPrefix + encodeURIComponent(t);
}

function isRecordable(url) {
  return /^https?:\/\//i.test(url || '');
}

/**
 * Runs inside a page (via executeJavaScript or the preload). Returns the main readable text,
 * skipping navigation, scripts and form controls. Values typed into inputs are never included:
 * innerText does not contain them.
 */
function extractReadable(maxChars) {
  const pick = document.querySelector('article, main, [role="main"]') || document.body;
  if (!pick) return { title: document.title, text: '', links: [] };
  const clone = pick.cloneNode(true);
  clone.querySelectorAll('script, style, noscript, nav, footer, header, aside, form, input, textarea, select, button, svg, iframe, [aria-hidden="true"]')
    .forEach((n) => n.remove());
  const host = document.createElement('div');
  host.style.cssText = 'position:fixed;left:-99999px;top:0;width:900px;';
  host.appendChild(clone);
  document.documentElement.appendChild(host);
  let text = clone.innerText || '';
  host.remove();
  text = text.replace(/\n{3,}/g, '\n\n').replace(/[ \t]{2,}/g, ' ').trim().slice(0, maxChars);
  const links = Array.from(pick.querySelectorAll('a[href]')).slice(0, 80)
    .map((a) => ({ text: (a.innerText || '').trim().slice(0, 120), href: a.href }))
    .filter((l) => l.text && /^https?:/.test(l.href));
  return { title: document.title, url: location.href, text, links };
}

/** Build a JS snippet that clicks the first element matching `selector`. */
function clickScript(selector) {
  return `(() => {
    const el = document.querySelector(${JSON.stringify(String(selector))});
    if (!el) return { ok: false, error: 'no element matches selector' };
    el.scrollIntoView({ block: 'center' });
    el.click();
    return { ok: true, tag: el.tagName.toLowerCase(), text: (el.innerText || el.value || '').slice(0, 80) };
  })()`;
}

/** Build a JS snippet that types into an input/textarea. Refuses password and hidden fields. */
function typeScript(selector, text, submit) {
  return `(() => {
    const el = document.querySelector(${JSON.stringify(String(selector))});
    if (!el) return { ok: false, error: 'no element matches selector' };
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type === 'password' || type === 'hidden' || el.autocomplete === 'cc-number' || /cc-|one-time-code/.test(el.autocomplete || ''))
      return { ok: false, error: 'refusing to type into a password, payment or one-time-code field' };
    if (!('value' in el) && !el.isContentEditable) return { ok: false, error: 'element is not editable' };
    el.focus();
    if (el.isContentEditable) el.textContent = ${JSON.stringify(String(text))};
    else {
      const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(String(text))});
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    if (${submit ? 'true' : 'false'}) {
      if (el.form && el.form.requestSubmit) el.form.requestSubmit();
      else el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    }
    return { ok: true };
  })()`;
}

/** Tiny router: match "GET /tabs/abc/read" against "GET /tabs/:id/read". */
function matchRoute(routes, method, pathname) {
  for (const r of routes) {
    if (r.method !== method) continue;
    const want = r.path.split('/');
    const got = pathname.split('/');
    if (want.length !== got.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < want.length; i++) {
      if (want[i].startsWith(':')) params[want[i].slice(1)] = decodeURIComponent(got[i]);
      else if (want[i] !== got[i]) { ok = false; break; }
    }
    if (ok) return { handler: r.handler, params };
  }
  return null;
}

/** Constant-time string compare for the auth token. */
function safeEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

module.exports = { searchQuery, resolveInput, isRecordable, extractReadable, clickScript, typeScript, matchRoute, safeEqual, DEFAULT_SEARCH };

'use strict';
const P = window.promptly;
const $ = (id) => document.getElementById(id);
let paused = false;

function li(title, url, snippet, why) {
  const el = document.createElement('li');
  const a = document.createElement('a');
  a.textContent = title || url;
  a.title = url;
  a.onclick = () => P.open_external(url);
  el.append(a);
  if (snippet) { const s = document.createElement('div'); s.className = 's'; s.textContent = snippet; el.append(s); }
  if (why) { const w = document.createElement('div'); w.className = 'why'; w.textContent = why; el.append(w); }
  return el;
}

function empty(text) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = text; return e; }

async function refresh() {
  const st = await P.mem_stats();
  if (!st) {
    $('status').textContent = 'Memory daemon offline';
    $('rec').className = 'rec off';
    $('stats').textContent = 'Run `promptly serve`. Browsing still works; nothing is recorded.';
    return;
  }
  paused = st.paused;
  $('rec').className = 'rec' + (paused ? ' off' : '');
  $('status').textContent = paused ? 'Recording paused' : 'Remembering, on this device only';
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  $('stats').textContent = `${st.pages} pages · ${st.visits} visits · ${st.events} signals · kept ${st.retention_days} days`;

  const ints = await P.mem_interests();
  const chips = (el, items, key) => {
    el.replaceChildren(...(items || []).slice(0, 10).map((i) => { const c = document.createElement('span'); c.className = 'chip'; c.textContent = i[key]; return c; }));
  };
  chips($('topics'), ints && ints.topics, 'topic');
  chips($('domains'), ints && ints.domains, 'domain');
  if (!ints || !ints.topics.length) $('topics').replaceChildren(empty('Nothing yet. It learns from what you read and how long you stay.'));

  const sug = (await P.mem_suggest()) || [];
  $('suggest').replaceChildren(...sug.map((f) => {
    const el = document.createElement('li');
    el.append(document.createTextNode(f.text + ' '));
    const b = document.createElement('button');
    b.className = 'pill'; b.textContent = 'Add to profile';
    b.onclick = async () => { await P.mem_accept(f); el.remove(); };
    el.append(b);
    return el;
  }));

  const rec = (await P.mem_recent()) || [];
  $('recent').replaceChildren(...(rec.length ? rec.map((r) => li(r.title, r.url, null, `${r.at.replace('T', ' ').slice(0, 16)} · ${r.dwell_s}s`)) : [empty('No visits yet.')]));
}

$('pause').onclick = async () => { await P.mem_pause(!paused); refresh(); };
$('go').onclick = async () => {
  const t = $('task').value.trim();
  if (!t) return;
  $('ctx').textContent = 'Building…';
  $('ctxwrap').hidden = false;
  $('ctx').textContent = (await P.mem_context(t)) || 'Memory daemon offline.';
};
$('task').onkeydown = (e) => { if (e.key === 'Enter') $('go').click(); };
$('copy').onclick = async () => { await P.clip_write($('ctx').textContent); $('copy').textContent = 'Copied'; setTimeout(() => ($('copy').textContent = 'Copy for any AI chat'), 1200); };

let qTimer;
$('q').oninput = () => {
  clearTimeout(qTimer);
  qTimer = setTimeout(async () => {
    const q = $('q').value.trim();
    if (!q) return $('hits').replaceChildren();
    const hits = (await P.mem_search(q)) || [];
    $('hits').replaceChildren(...(hits.length ? hits.map((h) => li(h.title, h.url, h.snippet, h.why.join(' · '))) : [empty('No matches.')]));
  }, 250);
};

const msg = (t) => { $('fmsg').textContent = t; setTimeout(() => ($('fmsg').textContent = ''), 3000); refresh(); };
$('fsite').onclick = async () => { const r = await P.mem_forgetSite(); msg(r ? `Forgot ${r.pages_removed} page(s).` : 'Nothing to forget here.'); };
$('block').onclick = async () => { const r = await P.mem_block(); msg(r ? 'This site will not be recorded again.' : 'Not a website.'); };
$('fhour').onclick = async () => { const r = await P.mem_forgetRecent(60); msg(r ? `Forgot ${r.pages_removed} page(s) from the last hour.` : 'Daemon offline.'); };

P.on('panel:refresh', refresh);
refresh();

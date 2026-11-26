'use strict';
const test = require('node:test');
const assert = require('node:assert');
const lib = require('../src/lib');

test('searchQuery extracts queries from search engines', () => {
  assert.strictEqual(lib.searchQuery('https://www.google.com/search?q=rust+async&hl=en'), 'rust async');
  assert.strictEqual(lib.searchQuery('https://duckduckgo.com/?q=vector%20db'), 'vector db');
  assert.strictEqual(lib.searchQuery('https://github.com/search?q=mcp+server'), 'mcp server');
  assert.strictEqual(lib.searchQuery('https://example.com/search?q=x'), null);
  assert.strictEqual(lib.searchQuery('not a url'), null);
});

test('resolveInput tells URLs from searches', () => {
  assert.strictEqual(lib.resolveInput('example.com'), 'https://example.com');
  assert.strictEqual(lib.resolveInput('localhost:3000/x'), 'http://localhost:3000/x');
  assert.strictEqual(lib.resolveInput('https://a.b/c'), 'https://a.b/c');
  assert.match(lib.resolveInput('how do vector indexes work'), /^https:\/\/duckduckgo\.com\/\?q=how%20do/);
  assert.strictEqual(lib.resolveInput('  '), null);
});

test('matchRoute extracts params', () => {
  const routes = [{ method: 'GET', path: '/tabs/:id/read', handler: 1 }, { method: 'GET', path: '/tabs', handler: 2 }];
  assert.deepStrictEqual(lib.matchRoute(routes, 'GET', '/tabs/ab12/read'), { handler: 1, params: { id: 'ab12' } });
  assert.strictEqual(lib.matchRoute(routes, 'GET', '/tabs').handler, 2);
  assert.strictEqual(lib.matchRoute(routes, 'POST', '/tabs'), null);
});

test('scripts embed selectors safely and refuse passwords', () => {
  const s = lib.typeScript('input"]); alert(1); //', 'hi', false);
  assert.ok(s.includes(JSON.stringify('input"]); alert(1); //')));
  assert.ok(s.includes("type === 'password'"));
  assert.ok(lib.clickScript('#a').includes('"#a"'));
});

test('safeEqual', () => {
  assert.ok(lib.safeEqual('abc', 'abc'));
  assert.ok(!lib.safeEqual('abc', 'abd'));
  assert.ok(!lib.safeEqual(undefined, 'abc'));
});

// Bridge for the toolbar and memory panel. Exposes a fixed list of calls; no raw ipc.
'use strict';
const { contextBridge, ipcRenderer } = require('electron');

const calls = [
  'tab:new', 'tab:close', 'tab:activate', 'tab:navigate', 'tab:back', 'tab:forward', 'tab:reload', 'tab:agentDrive',
  'panel:toggle', 'window:private', 'state:get',
  'mem:stats', 'mem:policy', 'mem:pause', 'mem:block', 'mem:forgetSite', 'mem:forgetRecent', 'mem:recent',
  'mem:interests', 'mem:search', 'mem:context', 'mem:suggest', 'mem:accept', 'clip:write', 'open:external',
];
const api = {};
for (const c of calls) api[c.replace(':', '_')] = (...args) => ipcRenderer.invoke(c, ...args);
api.on = (channel, cb) => {
  if (!['state', 'focus-address', 'panel:refresh'].includes(channel)) return;
  ipcRenderer.on(channel, (_e, data) => cb(data));
};
contextBridge.exposeInMainWorld('promptly', api);

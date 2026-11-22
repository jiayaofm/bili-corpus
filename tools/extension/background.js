// 本地服务的代理。
// 为什么不在 content script 里直接 fetch：content script 受页面 CORS 约束，
// 而 service worker 有 host_permissions，可以跨域访问 127.0.0.1。

const BASE = 'http://127.0.0.1:8849';
// 改写是串行的（服务端一次只跑一个生成），连点多个按钮时后面的要排队。
// 实测单条 0.8~2.7 秒，排到第 5 个就要十几秒，所以给足 90 秒。
const TIMEOUT = 90000;

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('超时 ' + ms + 'ms')), ms)),
  ]);
}

async function call(path, body) {
  const opt = body
    ? {
        method: 'POST',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify(body),
      }
    : { method: 'GET' };
  const r = await withTimeout(fetch(BASE + path, opt), TIMEOUT);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
  return data;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    if (msg.type === 'health') return call('/health');
    if (msg.type === 'score') return call('/score', { texts: msg.texts });
    if (msg.type === 'rewrite') return call('/rewrite', { text: msg.text });
    throw new Error('未知消息类型: ' + msg.type);
  })()
    .then(sendResponse)
    .catch((e) => sendResponse({ error: String(e && e.message ? e.message : e) }));
  return true; // 异步响应
});

// 点扩展图标 = 开/关徽标显示
chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: 'toggle' });
  } catch (e) {
    // 页面不是 B站视频页，忽略
  }
});

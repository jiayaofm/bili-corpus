// B站评论 AI 味检测 —— 页面脚本
//
// 干三件事：
//   1. 找出评论区里的每一条评论正文
//   2. 批量送给本地服务打分（通用 AI 腔 + 鱼腥味）
//   3. 在评论前挂徽标；鱼腥味高的给一个「改写」按钮
//
// 设计上的两条底线：
//   · 绝不修改页面原有的评论内容。改写结果放在单独的块里，原文保留。
//   · 找不到本地服务时静默降级，不弹窗、不刷屏。
//
// ## 为什么找评论这么麻烦
//
// B站 新评论区是 Web Component，正文藏在好几层 Shadow DOM 里，大致是：
//
//   bili-comments
//     #shadow-root
//       bili-comment-thread-renderer
//         #shadow-root
//           bili-comment-renderer
//             #shadow-root
//               bili-rich-text
//                 #shadow-root
//                   #contents > span   ← 真正的正文
//
// 所以不能只 querySelector，必须穿透 shadow root；
// 也不认死某一个类名（改版就废），而是三条路子一起上：
//   ① 已知选择器
//   ② rich-text 类组件里的 #contents
//   ③ 「在评论区上下文里的叶子文本节点」兜底
//
// 调试：控制台执行 __dshDebug() 会打印找到了什么、为什么没认出来。

(() => {
  'use strict';

  const BATCH = 20;
  const MIN_LEN = 3;
  const MAX_LEN = 300;
  const SCAN_LIMIT = 150;
  const ROOT_CAP = 200;

  let enabled = true;
  let scanning = false;
  let stats = { scored: 0, high: 0 };

  const done = new WeakSet();      // 已处理过的元素
  const seenText = new Set();      // 已打过分的内容
  const queue = [];
  const wraps = new Set();        // 已注入的徽标行，用于整体隐藏/显示
  let flushTimer = null;

  // ---------------------------------------------------------------- 穿透 shadow
  function collectRoots(root, out, depth) {
    if (!root || out.length >= ROOT_CAP) return out;
    out.push(root);
    if (depth > 12) return out;
    let all;
    try {
      all = root.querySelectorAll('*');
    } catch (e) {
      return out;
    }
    for (const el of all) {
      if (out.length >= ROOT_CAP) break;
      if (el.shadowRoot) collectRoots(el.shadowRoot, out, depth + 1);
    }
    return out;
  }

  // 从任意节点往上爬，穿过 shadow 边界，看是不是在评论区里
  function inCommentContext(el) {
    let n = el;
    for (let d = 0; d < 40 && n; d++) {
      const tag = (n.tagName || '').toLowerCase();
      const cls = typeof n.className === 'string' ? n.className : '';
      const id = n.id || '';
      if (/comment|reply|rich-text|richtext/.test(tag + ' ' + cls + ' ' + id)) return true;
      try {
        const r = n.getRootNode && n.getRootNode();
        if (r && r.host) { n = r.host; continue; }
      } catch (e) { /* ignore */ }
      n = n.parentElement;
    }
    return false;
  }

  // ---------------------------------------------------------------- 候选
  const KNOWN = [
    '.reply-content',
    '[class*="reply-content"]',
    '[class*="content-warp"]',
    '.comment-content',
    '[class*="comment-content"]',
    '.reply-context',
  ];

  // 剥掉「回复 @某人:」前缀。
  // 不剥的话，@、冒号、用户名都会进特征——实测同一句话
  // 「回复 @我是蓝色大肥鱼: 插个队，顺便我来帮你写」
  // 带前缀 94.0%，剥掉后 23.7%，差 70 个点。
  // 训练数据里一条前缀都没有（标注集 0%，语料 0%），所以这是纯推理期 bug。
  const REPLY_PREFIX = /^回复\s*@[^:：\n]{1,40}[:：]\s*/;

  function cleanText(el) {
    let t = (el.innerText || el.textContent || '').trim();
    t = t.replace(/\s+/g, ' ');
    t = t.replace(REPLY_PREFIX, '');
    t = t.replace(/^@[^\s:：]{1,40}\s+/, '');      // 行首 @某人 + 空格
    t = t.replace(/\s*(展开|收起|点击展开|回复)\s*$/, '');
    return t.trim();
  }

  function isSane(t) {
    if (t.length < MIN_LEN || t.length > MAX_LEN) return false;
    // 纯时间 / 纯数字 / 明显是按钮文字
    if (/^[\d\s:：\-年月日秒分小时]+$/.test(t)) return false;
    if (/^(回复|展开|收起|点赞|点踩|分享|举报|删除|置顶|热门|最新|按热度|按时间)$/.test(t)) return false;
    return true;
  }

  function addCandidate(el, out, seen) {
    if (!el || seen.has(el)) return;
    const t = cleanText(el);
    if (!isSane(t)) return;
    seen.add(el);
    out.push({ el, text: t });
  }

  function findCandidates() {
    const roots = collectRoots(document, [], 0);
    const out = [];
    const seen = new Set();

    // ① 已知选择器
    for (const r of roots) {
      for (const sel of KNOWN) {
        let list;
        try { list = r.querySelectorAll(sel); } catch (e) { continue; }
        for (const el of list) addCandidate(el, out, seen);
      }
    }
    if (out.length) return { list: out, roots: roots.length, how: '已知选择器' };

    // ② rich-text 组件里的 #contents（B站新版正文就在这）
    for (const r of roots) {
      let list;
      try { list = r.querySelectorAll('#contents, [id="contents"]'); } catch (e) { continue; }
      for (const el of list) {
        if (!inCommentContext(el)) continue;
        addCandidate(el, out, seen);
      }
    }
    if (out.length) return { list: out, roots: roots.length, how: 'rich-text #contents' };

    // ③ 兜底：评论区上下文里的叶子文本容器
    for (const r of roots) {
      let all;
      try { all = r.querySelectorAll('*'); } catch (e) { continue; }
      for (const el of all) {
        if (el.children && el.children.length > 0) continue;   // 只看叶子
        if (!inCommentContext(el)) continue;
        addCandidate(el, out, seen);
        if (out.length >= SCAN_LIMIT) break;
      }
      if (out.length >= SCAN_LIMIT) break;
    }
    return { list: out, roots: roots.length, how: out.length ? '叶子文本兜底' : '没找到' };
  }

  // ---------------------------------------------------------------- 徽标
  // 样式一律用内联写死。
  // 原因：B站 的评论在 Shadow DOM 里，manifest 注入的 content.css 只作用于
  // 普通文档，穿不进 shadow 边界——所以之前所有徽标都是裸文字，一个颜色都没有。
  const S = {
    badges: 'display:inline-flex;gap:4px;align-items:center',
    badge: 'display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;' +
           'font-weight:600;color:#fff;cursor:help;white-space:nowrap;opacity:.92',
    lv: 'font-size:10px;color:#8a93a0;border:1px solid #d5dae1;border-radius:6px;' +
        'padding:0 4px;white-space:nowrap',
  };
  const AI_BG = { low: '#4a9e6a', medium: '#c99021', high: '#c0392b' };
  const FISH_BG = { low: '#4a7fa5', medium: '#3a6fa0', high: '#23558c', na: '#c8ccd2' };

  // 从评论容器里找「LV6」这种等级标记，找不到就算了（不参与打分）
  function findLevel(el) {
    try {
      let n = el;
      for (let d = 0; d < 8 && n; d++) {
        const root = n.getRootNode && n.getRootNode();
        const scope = (root && root.host) ? root : (n.parentElement || document);
        const hits = scope.querySelectorAll && scope.querySelectorAll('[class*="level"],[class*="Level"]');
        if (hits) {
          for (const h of hits) {
            const m = (h.textContent || '').match(/LV\s*\d/i);
            if (m) return m[0].toUpperCase().replace(/\s+/g, '');
          }
        }
        n = n.parentElement || (root && root.host);
        if (!n) break;
      }
    } catch (e) { /* 拿不到就算了 */ }
    return null;
  }

  // 找评论作者名（用于统计「同一人说了几次」）。
  // 这是唯一能落地的 ID 信号：AI 评论号最典型的形态是同一账号反复刷，
  // 而「低等级更可能是 AI」我们手上没有数据能验证（93% 的语料没等级）。
  function findAuthor(el) {
    try {
      let n = el;
      for (let d = 0; d < 8 && n; d++) {
        const root = n.getRootNode && n.getRootNode();
        const scope = (root && root.host) ? root : (n.parentElement || document);
        const hits = scope.querySelectorAll &&
                     scope.querySelectorAll('[class*="user-name"],[class*="author"],[class*="nickname"]');
        if (hits) {
          for (const h of hits) {
            const t = (h.textContent || '').trim();
            if (t && t.length <= 30) return t;
          }
        }
        n = n.parentElement || (root && root.host);
        if (!n) break;
      }
    } catch (e) { /* 拿不到就算了 */ }
    return null;
  }

  // 每个账号的「签名片段」记录：{ frags: Map(片段->次数), wraps: [徽标行] }
  const authorSig = new Map();

  // 纯标点/空白的片段不算签名，否则「，，，」也会被当成特征
  const PUNCT_ONLY = /^[\s，。！？、；：…—,.!?;:~()（）【】\[\]"'“”‘’·]*$/;

  // 抽出「非中文片段」：颜文字、表情标记、固定后缀。
  // 例：(ˉ▽￣～)  [doge]  😂😂😂
  function signatureFragments(text) {
    const out = new Set();
    const re = /[^\u4e00-\u9fff0-9A-Za-z]{3,20}/g;
    let m;
    while ((m = re.exec(text))) {
      const f = m[0].trim();
      if (f.length < 3 || PUNCT_ONLY.test(f)) continue;
      out.add(f);
    }
    return out;
  }

  function addSigTag(wrap, text, tip, warn) {
    if (!wrap || wrap.querySelector('[data-dsh-sig]')) return;
    const t = document.createElement('span');
    t.dataset.dshSig = '1';
    t.style.cssText = S.lv + (warn
      ? ';color:#7a3b00;border-color:#e0b98a;background:#fdf6ec'
      : ';color:#6a3d8f;border-color:#c9b3e0;background:#f7f2fc');
    t.textContent = text;
    t.title = tip;
    wrap.append(t);
  }

  // 同一账号的几条评论带着同样的非中文片段 → 这是人设号最强的指纹之一。
  // 它不依赖「写得像不像人」，而依赖「一个人不会每次都带同一串符号」。
  function noteAuthor(author, text, wrap) {
    if (!author || !wrap) return;
    let rec = authorSig.get(author);
    if (!rec) { rec = { frags: new Map(), wraps: [] }; authorSig.set(author, rec); }
    if (rec.wraps.indexOf(wrap) < 0) rec.wraps.push(wrap);
    let hit = null;
    for (const f of signatureFragments(text)) {
      const n = (rec.frags.get(f) || 0) + 1;
      rec.frags.set(f, n);
      if (n >= 2 && !hit) hit = f;
    }
    if (hit && rec.wraps.length >= 2) {
      for (const w of rec.wraps) {
        addSigTag(w, '同款 ' + hit,
          '这个账号的 ' + rec.wraps.length + ' 条评论里都带着同样的「' + hit + '」\n' +
          '人设号常见特征：固定颜文字 / 固定后缀反复出现');
      }
    }
    // 同一人刷屏
    if (rec.wraps.length >= 3) {
      for (const w of rec.wraps) {
        addSigTag(w, '刷屏 ×' + rec.wraps.length,
          '这个账号在本页已经发了 ' + rec.wraps.length + ' 条评论\n' +
          'AI 评论号最典型的形态就是同一账号反复刷', true);
      }
    }
  }

  function makeBadge(res, level, author) {
    const b = document.createElement('span');
    b.style.cssText = S.badges;

    const ai = document.createElement('span');
    ai.style.cssText = S.badge + ';background:' + (AI_BG[res.ai_level] || AI_BG.medium);
    const aiPct = res.ai_prob * 100;
    ai.textContent = 'AI ' + (aiPct >= 10 ? Math.round(aiPct) : aiPct.toFixed(1)) + '%';
    ai.title = '通用 AI 腔：长句、书面语、复述标题、最后升华\n' +
               '颜色按「在真人里排多少分位」，不按概率绝对值：\n' +
               '真人自身中位只有 4.6%，所以 70% 已经排到 91 分位——少见，但真人也有\n' +
               '绿 <75 分位　黄 75~95　红 ≥95\n' +
               '⚠️ 对「长句子、讲道理、带链接」的真人评论容易判高';
    b.append(ai);

    // 鱼腥味徽标的显示条件：AI 判高，或者鱼分位 ≥90。
    //
    // 只挂「鱼 62」这种普通分数是噪声——真人自身的分位本来就均匀分布在 0~100。
    // 但也不能只在 AI 高时才显示：AI 分类器对长句说理的真人会误判高，
    // 而真正的鱼腔有时 AI 分并不高。鱼分位 ≥90 是更可靠的信号，单独也要显示。
    const fishy = res.fish_pct != null && res.fish_pct >= 90;
    if (res.ai_level === 'high' || fishy) {
      const pct = res.fish_pct;
      const fish = document.createElement('span');
      if (pct == null) {
        fish.style.cssText = S.badge + ';background:' + FISH_BG.na + ';opacity:.7';
        fish.textContent = '鱼 —';
        fish.title = '评论太短（不足 10 字），鱼腥味测不出来\n' +
                     '训练数据里几乎没有 10 字以下的真人样本，打分没有依据';
      } else {
        fish.style.cssText = S.badge + ';background:' + (FISH_BG[res.fish_level] || FISH_BG.medium);
        fish.textContent = '鱼 ' + Math.round(pct);
        fish.title = '鱼腥味：比 ' + Math.round(pct) + '% 的真人评论更重\n' +
                     '真人自身的分位是均匀分布的，50~70 都算正常\n' +
                     '低 <75　中 75~90　高 ≥90\n' +
                     '⚠️ 这个分数可以被「换标点加表情」骗到，仅供参考';
      }
      b.append(fish);
    }

    // 账号名本身就是线索：「我是蓝色大肥鱼」「暗灵家的大肥鱼」这类。
    // 只做展示，不进分数——名字不是文本特征，混进模型里会变成作弊。
    if (author && /(大肥鱼|肥鱼|本鱼|鱼尚在|鲦鱼|鲸鱼娘|机器里的鱼)/.test(author)) {
      const tag = document.createElement('span');
      tag.style.cssText = S.lv + ';color:#7a3b00;border-color:#e0b98a;background:#fdf6ec';
      tag.textContent = '名含鱼';
      tag.title = '账号名里有「鱼」相关字样：' + author + '\n' +
                  '⚠️ 只根据名字判断，没有看内容，不代表这条评论就是 AI 写的';
      b.append(tag);
    }

    if (level) {
      const lv = document.createElement('span');
      lv.style.cssText = S.lv;
      lv.textContent = level;
      lv.title = '账号等级（仅作参考，不参与打分）\n' +
                 '⚠️ 我们手上 93% 的语料没有等级数据，无法验证「低等级更可能是 AI」';
      b.append(lv);
    }
    return b;
  }

  function makeRewriteBtn(text) {
    const btn = document.createElement('button');
    btn.style.cssText =
      'border:1px solid #3a6fa0;background:#fff;color:#23558c;border-radius:8px;' +
      'font-size:11px;padding:1px 8px;cursor:pointer;line-height:1.5';
    btn.textContent = '改成人话';
    btn.addEventListener('click', async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (btn.disabled) return;
      btn.disabled = true;
      btn.textContent = '改写中…';
      const holder = document.createElement('div');
      holder.style.cssText =
        'margin:4px 0 6px;padding:8px 10px;border-left:3px solid #3a6fa0;' +
        'background:#f6f9fc;border-radius:0 6px 6px 0;font-size:13px;' +
        'line-height:1.6;color:#333';
      holder.textContent = '（等本地模型，连点多个会合并处理，通常几秒）';
      btn.after(holder);
      try {
        const r = await send({ type: 'rewrite', text });
        if (r.error) throw new Error(r.error);
        holder.textContent = '';
        const out = document.createElement('div');
        out.textContent = r.rewritten;
        const meta = document.createElement('div');
        meta.style.cssText = 'margin-top:4px;font-size:11px;color:#667';
        // 统一用百分位显示：概率的绝对值在这两个检测器上都没有直接含义
        const d = (a, b, name) => {
          if (a == null || b == null) return name + ' —';
          const arrow = b <= a ? '↓' : '↑';
          return name + ' ' + Math.round(a) + ' → ' + Math.round(b) + ' ' + arrow;
        };
        meta.textContent =
          d(r.ai_before_pct, r.ai_after_pct, 'AI') + '　' +
          d(r.fish_before, r.fish_after, '鱼');
        holder.append(out, meta);
        btn.textContent = '已改写';
      } catch (e) {
        holder.textContent = '改写失败：' + (e.message || e);
        btn.disabled = false;
        btn.textContent = '重试';
      }
    });
    return btn;
  }

  // ---------------------------------------------------------------- 状态条
  let pill = null;
  function ensurePill() {
    if (pill && document.body.contains(pill)) return pill;
    pill = document.createElement('div');
    pill.className = 'dsh-pill';
    pill.innerHTML =
      '<b>评论 AI 味</b><span class="dsh-pill-stat">连接中…</span>' +
      '<button class="dsh-pill-tg" type="button">隐藏</button>';
    pill.querySelector('.dsh-pill-tg').addEventListener('click', () => {
      enabled = !enabled;
      pill.querySelector('.dsh-pill-tg').textContent = enabled ? '隐藏' : '显示';
      for (const w of wraps) w.style.display = enabled ? 'flex' : 'none';
    });
    document.body.appendChild(pill);
    return pill;
  }

  function setStat(txt, cls) {
    if (!pill) return;
    const s = pill.querySelector('.dsh-pill-stat');
    s.textContent = txt;
    s.className = 'dsh-pill-stat ' + (cls || '');
  }

  // ---------------------------------------------------------------- 通信
  function send(msg) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(msg, (resp) => {
        if (chrome.runtime.lastError) resolve({ error: chrome.runtime.lastError.message });
        else resolve(resp || { error: '空响应' });
      });
    });
  }

  // ---------------------------------------------------------------- 打分
  function enqueue(el, text) {
    if (seenText.has(text)) { done.add(el); return; }
    if (queue.some((q) => q.text === text)) return;
    queue.push({ el, text });
    if (queue.length >= BATCH) flush();
    else if (!flushTimer) flushTimer = setTimeout(flush, 400);
  }

  async function flush() {
    flushTimer = null;
    if (!queue.length) return;
    const batch = queue.splice(0, BATCH);
    const texts = batch.map((b) => b.text);
    texts.forEach((t) => seenText.add(t));
    const r = await send({ type: 'score', texts });
    if (r.error) {
      setStat('服务未连接（' + String(r.error).slice(0, 36) + '）', 'dsh-err');
      texts.forEach((t) => seenText.delete(t));
      batch.forEach((b) => queue.push(b));
      return;
    }
    const results = r.results || [];
    for (let i = 0; i < batch.length && i < results.length; i++) {
      const { el } = batch[i];
      const res = results[i];
      done.add(el);
      stats.scored++;
      if (res.ai_level === 'high') stats.high++;
      try {
        const wrap = document.createElement('div');
        // 同样的道理：段落容器在 shadow DOM 里，样式只能内联
        wrap.style.cssText =
          'display:flex;align-items:center;gap:6px;margin:2px 0 4px;' +
          'flex-wrap:wrap;font-size:12px;line-height:1.4';
        const author = findAuthor(el);
        wrap.append(makeBadge(res, findLevel(el), author));
        // 改写按钮只在「鱼腥味高」时出现，不看 AI 分。
        //
        // 为什么：模型训练的是「鱼腔 → 真人」。给它一条本来就是真人写的、
        // 只是比较长或说理的评论（AI 分会误判高），它无事可做，只能改几个字，
        // 用户看到的就是「基本没变化」——实测 AI 97% → 95%。
        // 真正的鱼腔（鱼分位 ≥90）才是它的训练域。
        const fishy = res.fish_pct != null && res.fish_pct >= 90;
        if (fishy) {
          wrap.append(makeRewriteBtn(batch[i].text));
        }
        // 徽标行插在评论正文元素的正上方，不动正文本身
        wraps.add(wrap);
        if (el.parentNode) el.parentNode.insertBefore(wrap, el);
        noteAuthor(author, batch[i].text, wrap);
      } catch (e) { /* 单条失败不影响其它 */ }
    }
    setStat('已标 ' + stats.scored + ' 条，其中高 AI 味 ' + stats.high + ' 条', '');
    if (queue.length) flush();
  }

  // ---------------------------------------------------------------- 扫描
  let lastHow = '';
  function scan() {
    if (scanning || !enabled) return;
    scanning = true;
    try {
      const { list, roots, how } = findCandidates();
      lastHow = how;
      let added = 0;
      for (const c of list) {
        if (added >= SCAN_LIMIT) break;
        if (done.has(c.el)) continue;
        enqueue(c.el, c.text);
        added++;
      }
      if (!stats.scored && how === '没找到') {
        setStat('没找到评论区（扫了 ' + roots + ' 个 shadow root）', 'dsh-err');
      }
    } finally {
      scanning = false;
    }
  }

  // 调试入口：控制台执行 __dshDebug()
  window.__dshDebug = () => {
    const { list, roots, how } = findCandidates();
    const info = {
      shadow_roots: roots,
      strategy: how,
      candidates: list.length,
      scored: stats.scored,
      samples: list.slice(0, 8).map((c) => ({
        tag: c.el.tagName,
        id: c.el.id || null,
        cls: (typeof c.el.className === 'string' ? c.el.className : '').slice(0, 60),
        len: c.text.length,
        text: c.text.slice(0, 40),
      })),
    };
    console.log('[DSH]', info);
    if (!list.length) {
      // 找不到时，把看起来像评论区的自定义元素列出来，方便定位
      const tags = new Set();
      for (const r of collectRoots(document, [], 0)) {
        let all;
        try { all = r.querySelectorAll('*'); } catch (e) { continue; }
        for (const el of all) {
          const t = (el.tagName || '').toLowerCase();
          if (t.includes('comment') || t.includes('reply') || t.includes('rich')) tags.add(t);
        }
      }
      console.log('[DSH] 页面上和评论相关的标签：', [...tags]);
    }
    return info;
  };

  // ---------------------------------------------------------------- 启动
  async function boot() {
    ensurePill();
    const h = await send({ type: 'health' });
    if (h.error) {
      setStat('本地服务未启动，请运行 comment_server.py', 'dsh-err');
      return;
    }
    setStat('服务已连接' + (h.lora ? '（改写可用）' : '（改写需 --lora）'), '');
    scan();
    let t = null;
    const kick = () => {
      if (t) return;
      t = setTimeout(() => { t = null; scan(); }, 700);
    };
    new MutationObserver(kick).observe(document.body, { childList: true, subtree: true });
    window.addEventListener('scroll', kick, { passive: true });
    let n = 0;
    const tick = setInterval(() => {
      scan();
      if (++n >= 12) clearInterval(tick);
    }, 2000);
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg && msg.type === 'toggle') {
      enabled = !enabled;
      const b = pill && pill.querySelector('.dsh-pill-tg');
      if (b) b.textContent = enabled ? '隐藏' : '显示';
      for (const w of wraps) w.style.display = enabled ? 'flex' : 'none';
      if (enabled) scan();
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();

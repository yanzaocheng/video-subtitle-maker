'use strict';
const $ = (id) => document.getElementById(id);
const csrf = document.querySelector('meta[name="asr-token"]').content;
const PROVIDER_IDS = ['siliconflow', 'deepinfra'];
let config = null;
let media = null;
let currentJob = null;
let busy = false;
let interactionPending = false;
let keyPending = false;
let cancelPending = false;
let cancelRequested = false;
let needsRefresh = false;
let pollTimer = null;
// Subtitle editor state: a working copy of the cues of the job shown in step 4.
let editorJobId = null;
let cues = [];
let dirty = false;
let savePending = false;
let overlayTrack = null;
let overlayCues = [];
let activeIndex = -1;
// Subtitle style: the working copy shown in the preview and the copy last saved by the backend.
let subtitleStyle = null;
let savedStyle = null;
const STYLE_KEYS = ['font', 'size', 'color', 'outline_color', 'border', 'position', 'margin'];
// Mosaic removal runs as its own job next to the subtitle job.
let demosaicJob = null;
let demosaicPending = false;
let demosaicTimer = null;
// Custom translation API form (saved by the backend; the key never comes back to the page).
let customPending = false;
let lastUntranslated = -1;
const FILE_KINDS = { srt: '目标 SRT', original: '原文 SRT', bilingual: '双语 SRT', ass: '带样式 ASS', bilingual_ass: '双语带样式 ASS', json: 'JSON' };

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if ((options.method || 'GET').toUpperCase() !== 'GET') headers.set('X-ASR-Token', csrf);
  const response = await fetch(path, { ...options, headers, credentials: 'same-origin', cache: 'no-store', mode: 'same-origin' });
  let data;
  try { data = await response.json(); }
  catch {
    const error = new Error(`本机服务返回了无法读取的响应（HTTP ${response.status}），请刷新检查任务状态。`);
    error.status = response.status;
    throw error;
  }
  if (!response.ok) {
    const error = new Error(data.error || `请求失败：HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}
function status(message, type = '') {
  $('status').textContent = message;
  $('status').className = type;
}
function requestRefresh() {
  needsRefresh = true;
  $('refresh-page').hidden = false;
}
function formatDuration(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0 || value == null) return '时长未知';
  const seconds = Math.round(number);
  return `${Math.floor(seconds / 3600).toString().padStart(2, '0')}:${Math.floor(seconds % 3600 / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}
function formatSize(value) {
  const size = Number(value);
  if (!Number.isFinite(size) || size < 0 || value == null) return '大小未知';
  return size >= 1073741824 ? `${(size / 1073741824).toFixed(2)} GB` : `${(size / 1048576).toFixed(2)} MB`;
}
function fillSelect(id, items, fallback, label = (item) => item.name) {
  if (!Array.isArray(items) || !items.length) return;
  const previous = $(id).value;
  $(id).replaceChildren();
  for (const item of items) {
    const option = document.createElement('option');
    option.value = item.code;
    option.textContent = label(item);
    option.disabled = item.available === false;
    $(id).append(option);
  }
  $(id).value = items.some((item) => item.code === previous && item.available !== false) ? previous : fallback;
  if (!$(id).value || $(id).selectedOptions[0]?.disabled) $(id).value = items.find((item) => item.available !== false)?.code || '';
}
function providerName(code) { return config?.providers?.find((p) => p.code === code)?.name || code; }
function engine() { return config?.asr_engines?.find((item) => item.code === $('asr-engine').value); }
function translationItem() { return config?.translation_models?.find((item) => item.code === $('translation-quality').value); }
function translating() { return $('target-language').value !== 'original'; }
function customSaved() { return config?.custom_translation || { configured: false, key_saved: false, base_url: '', model: '', api_format: 'chat' }; }
function customSelected() { return $('translation-quality').value === 'custom'; }
function customForm() {
  return { base_url: $('custom-base-url').value.trim(), model: $('custom-model').value.trim(), api_format: $('custom-format').value || 'chat', api_key: $('custom-key').value.trim() };
}
function customDirty() {
  const form = customForm(), saved = customSaved();
  return form.base_url !== (saved.base_url || '') || form.model !== (saved.model || '') || form.api_format !== (saved.api_format || 'chat') || !!form.api_key;
}
function customTarget(value, format = 'chat') {
  // Mirrors custom_endpoint() of the backend, for the hint under the form only.
  try {
    const url = new URL(value.trim());
    if (!['http:', 'https:'].includes(url.protocol)) return null;
    const paths = (config?.custom_formats || []).map((item) => item.path);
    const suffix = (config?.custom_formats || []).find((item) => item.code === format)?.path || '/chat/completions';
    let path = url.pathname.replace(/\/+$/, '');
    if (!path.endsWith(suffix)) {
      const other = paths.find((item) => path.endsWith(item)) || '';
      path = (path.slice(0, path.length - other.length) || '/v1') + suffix;
    }
    const local = /^(localhost$|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|\[::1\]$)/.test(url.hostname);
    return { url: `${url.origin}${path}${url.search}`, host: url.hostname, exposed: url.protocol === 'http:' && !local };
  } catch { return null; }
}
function translationDestination() {
  const item = translationItem();
  if (item?.provider !== 'custom') return providerName(item?.provider);
  const target = customTarget(customSaved().base_url || '');
  return `自定义接口${target ? `（${target.host}）` : ''}`;
}
function customBlocked() {
  // The job uses what is saved, so a form that differs from it has to be saved first.
  if (!translating() || !customSelected()) return '';
  if (customDirty()) return '自定义 API 的内容还没有保存，请先在第 2 步点“保存”。';
  return customSaved().configured ? '' : '请先在第 2 步填写并保存自定义 API 的接口地址和模型名称。';
}
function requiredProviders() {
  const needed = new Set();
  const current = engine();
  if (current && current.provider !== 'local') needed.add(current.provider);
  if (translating() && translationItem() && translationItem().provider !== 'custom') needed.add(translationItem().provider);
  return [...needed];
}
function missingKeys() { return requiredProviders().filter((provider) => !config?.keys?.[provider]); }
function processingRange() {
  // The whole video is always processed.
  const total = media?.duration_seconds == null ? NaN : Number(media.duration_seconds);
  const known = Number.isFinite(total) && total > 0;
  return { effective: known ? total : null, total: known ? total : null };
}
function languageName() { return $('target-language').selectedOptions[0]?.textContent || '所选语言'; }
function costDescription(range) {
  const current = engine();
  let asr;
  if (current?.provider === 'local') asr = '本地 GPU 识别不产生识别费用';
  else if (range.effective == null) asr = '时长未知，无法估算识别费用';
  else asr = `按完整处理时长估算，识别费用约 ¥${(range.effective * (current?.price_per_second ?? 0.00022)).toFixed(4)}${current?.provider === 'deepinfra' ? '（按 1 美元≈7.2 元换算）' : ''}`;
  const translation = !translating() ? '原文模式不产生翻译请求。'
    : customSelected() ? `翻译费用由你的自定义接口决定（${customSaved().model || '未设置模型'}；本机模型不收费）。`
    : `翻译另按 token 计费（${translationItem()?.name || '所选模型'}），提交前无法准确预估。`;
  return `${asr}；${translation}实际只处理检测到的语音，最终以服务商账单为准。`;
}
function updateEngineNote() {
  const current = engine();
  if (!current) { $('asr-note').textContent = ''; return; }
  if (current.provider === 'local') {
    const gpu = config.gpu;
    $('asr-note').textContent = current.available
      ? `免费，音频不离开本机。${gpu ? `显卡：${gpu.name}（${Math.round(gpu.total_mb / 1024)} GB）。` : ''}开始时加载模型约 20～60 秒；显存不足会自动减小批量，仍不足会提示改用 0.6B 或云端。${demosaicRunning() ? '去马赛克正在使用显卡：本地识别会等它完成后才开始，着急可改用云端识别。' : ''}`
      : `${current.reason}，暂不可用。`;
  } else {
    const saved = config.keys?.[current.provider];
    $('asr-note').textContent = `语音片段会发送到${providerName(current.provider)}识别并计费。${saved ? '' : `尚未保存${providerName(current.provider)}的 Key。`}`;
  }
}
function updateRange() {
  const range = processingRange();
  $('range-summary').textContent = media ? `全部视频${range.effective == null ? '' : ` · 约 ${formatDuration(range.effective)}`}` : '';
  $('cost').textContent = media ? costDescription(range) : '选择视频后显示费用估算。';
  if (!translating()) $('bilingual').checked = false;
  $('translation-note').textContent = translating()
    ? `${$('source-language').selectedOptions[0]?.textContent || '原文'} → ${languageName()}。翻译会结合上下文纠正同音误识别；字幕文本会发送到${translationDestination()}。个别句子被翻译服务拒绝时会保留原文并标出，不影响其余字幕。`
    : '仅生成原文，不提交翻译。';
  updateEngineNote();
  updateControls();
}
function updateControls() {
  const ready = !!config;
  const desktop = config?.desktop_mode === true;
  const locked = !ready || busy || interactionPending || needsRefresh;
  const current = engine();
  const missing = ready ? missingKeys() : [];
  const localBlocked = current?.provider === 'local' && !current.available;
  const customIssue = ready ? customBlocked() : '';
  $('generate').disabled = locked || keyPending || customPending || !desktop || !media || !current || localBlocked || missing.length > 0 || !!customIssue;
  $('generate').textContent = busy ? '正在生成字幕…' : '生成字幕';
  for (const id of ['pick-media', 'asr-engine', 'source-language', 'target-language']) $(id).disabled = locked || !desktop;
  $('translation-quality').disabled = locked || !desktop || !translating();
  $('context-hint').disabled = locked || !desktop || !translating();
  $('skip-nonverbal').disabled = locked || !desktop || $('source-language').value !== 'ja';
  $('bilingual').disabled = locked || !desktop || !translating();
  for (const provider of PROVIDER_IDS) {
    for (const id of [`api-key-${provider}`, `save-key-${provider}`, `clear-key-${provider}`]) $(id).disabled = !ready || busy || interactionPending || keyPending || needsRefresh;
    $(`key-status-${provider}`).textContent = config?.keys?.[provider] ? '已安全保存到本机' : ready ? '尚未设置' : '';
  }
  const saved = PROVIDER_IDS.filter((provider) => config?.keys?.[provider]).map(providerName);
  $('key-status').textContent = ready ? (saved.length ? `已保存：${saved.join('、')}` : '尚未设置') : '正在读取…';
  $('generate-hint').textContent = busy || !ready ? '' : !media ? '请先选择视频。'
    : missing.length ? `请先在下方“API Key 设置”保存：${missing.map(providerName).join('、')}` : customIssue;
  updateCustom();
  $('cancel-job').disabled = !busy || !currentJob || cancelPending || cancelRequested || needsRefresh;
  $('cancel-job').textContent = cancelRequested ? '已请求取消，等待停止…' : cancelPending ? '正在请求取消…' : '取消任务';
  updateDemosaicControls();
}
function updateCustom() {
  const saved = customSaved(), form = customForm();
  $('custom-api').hidden = !customSelected();
  const locked = !config || busy || interactionPending || customPending || needsRefresh;
  for (const id of ['custom-base-url', 'custom-format', 'custom-model', 'custom-key']) $(id).disabled = locked;
  const filled = !!form.base_url && !!form.model;
  $('custom-save').disabled = locked || !filled || !customDirty();
  $('custom-test').disabled = locked || !filled;
  $('custom-clear').disabled = locked || (!saved.configured && !saved.key_saved);
  $('custom-key').placeholder = saved.key_saved ? '已保存 Key；留空表示不修改' : '本机模型可以留空';
  const target = customTarget(form.base_url, form.api_format);
  $('custom-note').textContent = !form.base_url ? '地址一般以 /v1 结尾；填写后这里会显示实际请求的地址。'
    : !target ? '地址需要以 http:// 或 https:// 开头。'
    : `实际请求：${target.url}${target.exposed ? '\n注意：这是未加密的 http 地址，Key 和字幕文本在网络上是明文传输的。' : ''}${customDirty() ? '\n尚未保存。' : saved.key_saved ? '\n已保存（含 Key）。' : '\n已保存（未设置 Key）。'}`;
}
function customMessage(text, type = '') {
  $('custom-message').textContent = text;
  $('custom-message').className = type;
}
function showCustom(data) {
  // The saved settings as the backend reports them; the key field is always emptied.
  config.custom_translation = data.custom_translation;
  config.translation_models = data.translation_models;
  fillSelect('translation-quality', config.translation_models, config.default_translation || 'pro');
  $('custom-base-url').value = data.custom_translation.base_url || '';
  $('custom-format').value = data.custom_translation.api_format || 'chat';
  $('custom-model').value = data.custom_translation.model || '';
  $('custom-key').value = '';
}
async function customCall(path, body, pending) {
  customPending = true;
  customMessage(pending);
  updateControls();
  try {
    return await api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } finally { customPending = false; }
}
function demosaicRunning() { return demosaicJob?.status === 'running'; }
function sampleSeconds() { return config?.demosaic?.sample_seconds || 60; }
function sampleStart() {
  // The sample starts where the step 1 preview is, moved back so that a full sample fits before the end.
  const player = $('video-preview');
  const total = Number(media?.duration_seconds) || player.duration || 0;
  const now = Number.isFinite(player.currentTime) ? player.currentTime : 0;
  return Math.max(0, Math.min(now, total - sampleSeconds()));
}
function updateDemosaicControls() {
  const info = config?.demosaic;
  const running = demosaicRunning();
  const video = media?.kind === 'video';
  const blocked = !info?.available || !video || running || demosaicPending || needsRefresh || interactionPending;
  $('demosaic-sample').disabled = blocked;
  $('demosaic-full').disabled = blocked;
  $('demosaic-sample').textContent = `试处理 ${sampleSeconds()} 秒`;
  $('demosaic-cancel').disabled = !running || demosaicPending || !!demosaicJob?.cancel_requested;
  $('demosaic-cancel').textContent = demosaicJob?.cancel_requested && running ? '正在取消…' : '取消去马赛克';
  $('demosaic-note').textContent = !config ? '正在检查去马赛克组件…'
    : !info?.available ? `${info?.reason || '去马赛克不可用'}（需要 NVIDIA RTX 20 系或更新的显卡），安装后重新打开客户端。`
    : `速度取决于显卡和分辨率，通常比实时播放慢（RTX 3070 处理 1080p 约 20～25 帧/秒，2 小时的视频约需 2.5～3 小时；4K 更慢）。建议先试处理 ${sampleSeconds()} 秒看看效果。`;
  $('demosaic-hint').textContent = !info?.available || running ? ''
    : !media ? '请先在第 1 步选择视频。'
    : !video ? '去马赛克只能处理视频，当前选择的是音频。'
    : `试处理从第 1 步预览的播放位置 ${formatDuration(sampleStart())} 开始，另存为单独的试看文件。`;
}
function demosaicMessage(text, type = '') {
  $('demosaic-message').textContent = text;
  $('demosaic-message').className = type;
}
function showDemosaic(job) {
  if (!job?.id) return;
  // A delayed response must not roll a finished snapshot back to running.
  if (demosaicJob?.id === job.id && demosaicJob.status !== 'running' && job.status === 'running') return;
  clearTimeout(demosaicTimer);
  const changed = demosaicJob?.id !== job.id || demosaicJob?.status !== job.status;
  demosaicJob = job;
  $('demosaic-progress-area').hidden = false;
  const progress = Number(job.progress);
  const percent = Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0;
  $('demosaic-progress').value = percent;
  $('demosaic-progress-value').textContent = `${Math.round(percent)}%`;
  $('demosaic-stage').textContent = job.stage || '正在处理';
  const range = job.sample ? `试处理 ${formatDuration(job.sample_start)} 起 ${Math.round(job.sample_seconds)} 秒` : '整个视频';
  $('demosaic-detail').textContent = `${job.filename} · ${range} · 已用时 ${formatDuration(job.elapsed_seconds)}`;
  $('demosaic-meta').textContent = { running: '进行中', done: '已完成', failed: '失败', cancelled: '已取消' }[job.status] || job.status;
  const done = job.status === 'done';
  $('demosaic-result').hidden = !done;
  if (changed) {
    $('demosaic-video').pause();
    if (done) $('demosaic-video').src = `/api/demosaic/${encodeURIComponent(job.id)}/media`;
    else $('demosaic-video').removeAttribute('src');
    $('demosaic-video').load();
  }
  if (changed) updateEngineNote();
  $('demosaic-files').replaceChildren();
  if (done) {
    for (const [label, path] of [['去码视频', job.output_path], ...(job.subtitle_files || []).map((path) => ['配套字幕', path])]) {
      const item = document.createElement('li');
      item.textContent = `${label} · ${path}`;
      $('demosaic-files').append(item);
    }
  }
  if (job.status === 'running') {
    demosaicMessage(job.cancel_requested ? '已请求取消，正在停止去马赛克程序…' : '去马赛克在后台进行，关闭窗口不会停止；重新打开客户端可以继续查看进度。');
    scheduleDemosaicPoll(job.id);
  } else if (done) {
    demosaicMessage(job.error || (job.sample ? '试处理完成，可以在下方预览或用播放器打开。' : '去马赛克完成，已另存为新视频。'), job.error ? 'error' : 'success');
    // Subtitle copies were added to the subtitle job's files.
    if (changed && job.subtitle_files?.length && currentJob && !busy) {
      api(`/api/jobs/${encodeURIComponent(currentJob.id)}`).then((data) => { if (currentJob?.id === data.job.id) showJob(data.job); }).catch(() => {});
    }
  } else {
    demosaicMessage(job.error || (job.status === 'cancelled' ? '已取消，未保存去码视频。' : '去马赛克失败。'), job.status === 'failed' ? 'error' : '');
  }
  updateControls();
}
function scheduleDemosaicPoll(id) {
  clearTimeout(demosaicTimer);
  demosaicTimer = setTimeout(async () => {
    try {
      const data = await api(`/api/demosaic/${encodeURIComponent(id)}`);
      if (demosaicJob?.id === id) showDemosaic(data.job);
    } catch (error) {
      if (demosaicJob?.id !== id) return;
      demosaicMessage(`暂时无法读取去马赛克进度：${error.message}`, 'error');
      if (error.status !== 404 && error.status !== 403) scheduleDemosaicPoll(id);
    }
  }, 2000);
}
async function startDemosaic(sample) {
  if (!media || demosaicPending || demosaicRunning() || $(sample ? 'demosaic-sample' : 'demosaic-full').disabled) return;
  const body = { media_id: media.id };
  if (sample) body.sample_start = Math.round(sampleStart() * 1000) / 1000;
  else {
    const stem = (media.name || '视频').replace(/\.[^.]*$/, '');
    if (!window.confirm(`处理整个视频：${media.name || ''}\n时长：${formatDuration(media.duration_seconds)}\n\n耗时较长（通常比视频本身更长），期间显卡被占用，本地 GPU 识别会排队等待。\n结果另存为“${stem}.restored.mp4”，原视频不改动。确认开始？`)) return;
  }
  demosaicPending = true;
  demosaicMessage('正在启动去马赛克…');
  updateControls();
  try {
    const data = await api('/api/demosaic', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    demosaicPending = false;
    showDemosaic(data.job);
  } catch (error) { demosaicMessage(error.message, 'error'); }
  finally { demosaicPending = false; updateControls(); }
}
function resetPreview() {
  for (const id of ['video-preview', 'audio-preview']) {
    $(id).pause();
    $(id).removeAttribute('src');
    $(id).load();
    $(id).hidden = true;
  }
  $('selected-path').textContent = media?.path || '尚未选择视频';
  $('selected-info').textContent = media ? `${media.kind === 'video' ? '视频' : '音频'} · ${formatDuration(media.duration_seconds)} · ${formatSize(media.size_bytes)}` : '—';
  if (media) {
    const player = $(media.kind === 'video' ? 'video-preview' : 'audio-preview');
    player.src = `/api/media/${encodeURIComponent(media.id)}/preview`;
    player.hidden = false;
    player.load();
  }
  updateRange();
}
function acceptMedia(item, message = '已按本机路径载入；不会上传或复制整个视频。') {
  if (!item?.id) throw new Error('本机服务没有返回有效的媒体 ID，未选择视频。');
  media = item;
  $('media-message').textContent = message;
  resetPreview();
}
async function pickMedia() {
  if (busy || interactionPending || needsRefresh || !config?.desktop_mode) return;
  interactionPending = true;
  updateControls();
  $('media-message').textContent = '请在 Windows 文件选择窗口中选择视频，可能需要切换到该窗口。';
  try {
    const data = await api('/api/media/pick', { method: 'POST' });
    if (data.media) acceptMedia(data.media);
    else $('media-message').textContent = media ? '已取消选择，保留原来的视频。' : '已取消选择。';
  } catch (error) { $('media-message').textContent = error.message; }
  finally { interactionPending = false; updateControls(); }
}
function pad(value, size = 2) { return String(value).padStart(size, '0'); }
function cueTime(seconds) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms % 3600000 / 60000))}:${pad(Math.floor(ms % 60000 / 1000))}.${pad(ms % 1000, 3)}`;
}
function translatedJob() { return currentJob?.target_language !== 'original'; }
function subtitleParts(cue) {
  const mode = $('overlay-mode').value;
  if (mode === 'off') return [];
  if (mode === 'original') return [{ text: translatedJob() ? cue.original_text || '' : cue.text }];
  const parts = [{ text: cue.text }];
  if (mode === 'bilingual' && translatedJob() && cue.original_text) parts.push({ text: cue.original_text, small: true });
  return parts;
}
function renderSubtitle() {
  const shown = overlayTrack ? [...(overlayTrack.activeCues || [])].map((item) => overlayCues.indexOf(item)).filter((index) => cues[index]).sort((a, b) => a - b) : [];
  const lines = [];
  for (const index of shown) {
    for (const part of subtitleParts(cues[index])) {
      if (!part.text.trim()) continue;
      const line = document.createElement('div');
      line.className = part.small ? 'subtitle-part small' : 'subtitle-part';
      const text = document.createElement('span');
      text.textContent = part.text.trim();
      line.append(text);
      lines.push(line);
    }
  }
  $('subtitle-block').replaceChildren(...lines);
}
function makeOverlayCue(cue) { return new VTTCue(cue.start, Math.max(cue.end, cue.start + 0.05), ''); }
function rebuildOverlay() {
  if (!overlayTrack) {
    // A hidden metadata track gives exact cue timing; renderSubtitle draws the text itself so it can be styled freely.
    overlayTrack = $('editor-video').addTextTrack('metadata', 'subtitle-preview');
    overlayTrack.mode = 'hidden';
    overlayTrack.addEventListener('cuechange', renderSubtitle);
  }
  for (const cue of [...(overlayTrack.cues || [])]) overlayTrack.removeCue(cue);
  overlayCues = cues.map((cue) => { const item = makeOverlayCue(cue); overlayTrack.addCue(item); return item; });
  renderSubtitle();
}
function refreshOverlayCue(index) {
  // Replacing the cue is the reliable way to apply new times.
  if (!overlayTrack || !overlayCues[index]) return;
  overlayTrack.removeCue(overlayCues[index]);
  overlayCues[index] = makeOverlayCue(cues[index]);
  overlayTrack.addCue(overlayCues[index]);
  renderSubtitle();
}
function layoutSubtitles() {
  // Cover exactly the picture (letterboxing in full screen included) so sizes and margins match the .ass export.
  const video = $('editor-video');
  const width = video.clientWidth, height = video.clientHeight;
  if (!width || !height) return;
  const sourceWidth = video.videoWidth || width, sourceHeight = video.videoHeight || height;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const pictureWidth = sourceWidth * scale, pictureHeight = sourceHeight * scale;
  const layer = $('subtitle-layer');
  layer.style.left = `${video.offsetLeft + (width - pictureWidth) / 2}px`;
  layer.style.top = `${video.offsetTop + (height - pictureHeight) / 2}px`;
  layer.style.width = `${pictureWidth}px`;
  layer.style.height = `${pictureHeight}px`;
  layer.style.setProperty('--unit', `${pictureHeight / 1080}px`);
}
function untranslatedIndexes() {
  // A line the translator turned down keeps its original text.
  if (!translatedJob()) return [];
  const indexes = [];
  cues.forEach((cue, index) => { if (cue.original_text?.trim() && cue.text.trim() === cue.original_text.trim()) indexes.push(index); });
  return indexes;
}
function styleChanged() { return !!subtitleStyle && !!savedStyle && STYLE_KEYS.some((key) => subtitleStyle[key] !== savedStyle[key]); }
function unsaved() { return dirty || styleChanged(); }
function showStyle(style) {
  $('style-font').value = style.font;
  $('style-size').value = style.size;
  $('style-color').value = style.color.toLowerCase();
  $('style-border').value = style.border;
  $('style-outline-color').value = style.outline_color.toLowerCase();
  $('style-position').value = style.position;
  $('style-margin').value = style.margin;
}
function readStyle() {
  return {
    font: $('style-font').value, size: Number($('style-size').value), color: $('style-color').value.toUpperCase(),
    outline_color: $('style-outline-color').value.toUpperCase(), border: $('style-border').value,
    position: $('style-position').value, margin: Number($('style-margin').value),
  };
}
function rgba(hex, alpha) {
  const value = parseInt(hex.slice(1), 16);
  return `rgba(${value >> 16 & 255}, ${value >> 8 & 255}, ${value & 255}, ${alpha})`;
}
function applyStyle() {
  const style = subtitleStyle;
  if (!style) return;
  const layer = $('subtitle-layer');
  const font = config?.subtitle_fonts?.find((item) => item.code === style.font);
  layer.style.fontFamily = `"${style.font}", "Microsoft YaHei", sans-serif`;
  layer.style.color = style.color;
  layer.style.setProperty('--sub-size', String(style.size));
  layer.style.setProperty('--sub-margin', String(style.margin));
  // Line spacing of ASS renderers is the font's cell height.
  layer.style.setProperty('--sub-line', String(font?.cell || 1.32));
  layer.style.setProperty('--sub-outline', style.outline_color);
  layer.style.setProperty('--sub-box', rgba(style.outline_color, 0.6));
  layer.dataset.border = style.border;
  layer.dataset.position = style.position;
  $('style-size-value').textContent = style.size;
  $('style-margin-value').textContent = style.margin;
  $('style-outline-label').textContent = style.border === 'box' ? '底框颜色' : '描边颜色';
  $('style-outline-color').disabled = style.border === 'none';
}
function setStyle(style, saved = false) {
  subtitleStyle = { ...style };
  if (saved) savedStyle = { ...style };
  showStyle(subtitleStyle);
  applyStyle();
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else $('video-stage').requestFullscreen().catch(() => { $('result-message').textContent = '无法进入全屏预览。'; });
}
function cueProblems() {
  const problems = [];
  cues.forEach((cue, index) => {
    if (!(cue.end - cue.start >= 0.05)) problems.push([index, '结束时间必须晚于开始时间']);
    else if (index > 0 && cue.start < cues[index - 1].end - 0.002) problems.push([index, '与上一条时间重叠']);
    if (!cue.text.trim()) problems.push([index, '字幕文字为空']);
  });
  return problems;
}
function updateEditorState() {
  const problems = cueProblems();
  const bad = new Set(problems.map(([index]) => index));
  const untranslated = new Set(untranslatedIndexes());
  for (const row of $('cue-list').children) {
    row.classList.toggle('invalid', bad.has(Number(row.dataset.index)));
    row.classList.toggle('untranslated', untranslated.has(Number(row.dataset.index)));
  }
  $('next-untranslated').hidden = !untranslated.size;
  $('next-untranslated').textContent = `未翻译 ${untranslated.size} 句 · 跳到下一句`;
  const pending = unsaved();
  $('edit-state').textContent = savePending ? '正在保存…'
    : problems.length ? `第 ${problems[0][0] + 1} 条：${problems[0][1]}${problems.length > 1 ? `（共 ${problems.length} 处问题）` : ''}`
    : dirty ? '有未保存的修改' : pending ? '字幕样式已修改，保存后写入 .ass 文件' : currentJob?.edited ? '已保存你的修改' : '';
  $('edit-state').className = problems.length ? 'muted edit-error' : 'muted';
  $('save-edits').disabled = !pending || problems.length > 0 || savePending || busy;
  $('revert-edits').disabled = !pending || savePending;
  $('restore-generated').hidden = !currentJob?.can_restore;
  $('restore-generated').disabled = savePending || busy;
}
function markDirty() { dirty = true; updateEditorState(); }
function textField(cls, field, value, label, placeholder = '') {
  const area = document.createElement('textarea');
  area.className = cls;
  area.dataset.field = field;
  area.value = value || '';
  area.rows = 1;
  area.spellcheck = false;
  area.placeholder = placeholder;
  area.setAttribute('aria-label', label);
  return area;
}
function actionButton(act, text, title, extra = '') {
  const button = document.createElement('button');
  button.dataset.act = act;
  button.textContent = text;
  button.title = title;
  if (extra) button.className = extra;
  return button;
}
function cueRow(cue, index) {
  const row = document.createElement('div');
  row.className = 'cue';
  row.dataset.index = index;
  const time = actionButton('seek', '', '从这句开始播放', 'cue-time');
  time.textContent = `#${index + 1}\n${cueTime(cue.start)}\n→ ${cueTime(cue.end)}`;
  const texts = document.createElement('div');
  texts.className = 'cue-texts';
  texts.append(textField('cue-text', 'text', cue.text, `第 ${index + 1} 条字幕`));
  if (translatedJob()) texts.append(textField('cue-original', 'original_text', cue.original_text, `第 ${index + 1} 条原文`, '原文（可留空）'));
  const actions = document.createElement('div');
  actions.className = 'cue-actions';
  actions.append(actionButton('start', '起点', '把开始时间设为当前播放位置'),
    actionButton('end', '终点', '把结束时间设为当前播放位置'),
    actionButton('delete', '删除', '删除这条字幕', 'subtle'));
  row.append(time, texts, actions);
  return row;
}
function renderCueList() {
  const list = $('cue-list');
  const scroll = list.scrollTop;
  list.replaceChildren(...cues.map(cueRow));
  list.scrollTop = scroll;
  activeIndex = -1;
}
function loadEditorCues(job) {
  cues = (job.cues || []).map((cue) => ({ ...cue }));
  dirty = false;
  lastUntranslated = -1;
  renderCueList();
  rebuildOverlay();
  updateEditorState();
}
function renderEditor(job) {
  const show = job.status !== 'running' && (job.cues?.length || 0) > 0;
  $('editor').hidden = !show;
  $('editor-empty').hidden = show;
  $('editor-empty').textContent = job.status === 'running' ? '字幕生成完成后，可以在这里边看视频边逐条修改。' : '这个任务没有可编辑的字幕。';
  if (!show) return;
  for (const option of $('overlay-mode').options) if (['bilingual', 'original'].includes(option.value)) option.disabled = job.target_language === 'original';
  if ($('overlay-mode').selectedOptions[0]?.disabled) $('overlay-mode').value = 'target';
  // Unsaved edits of the same job survive progress refreshes.
  if (editorJobId === job.id && dirty) { updateEditorState(); return; }
  if (editorJobId !== job.id) {
    $('editor-video').src = `/api/jobs/${encodeURIComponent(job.id)}/media`;
    $('editor-video').load();
    editorJobId = job.id;
  }
  loadEditorCues(job);
}
function findActiveCue(time) {
  let low = 0, high = cues.length - 1, found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (cues[middle].start <= time) { found = middle; low = middle + 1; } else high = middle - 1;
  }
  return found >= 0 && time <= cues[found].end ? found : -1;
}
async function saveEdits() {
  if ($('save-edits').disabled || !currentJob) return;
  savePending = true;
  updateEditorState();
  try {
    const payload = { cues: cues.map((cue) => ({ start: cue.start, end: cue.end, text: cue.text, original_text: cue.original_text || '' })), style: subtitleStyle };
    const data = await api(`/api/jobs/${encodeURIComponent(currentJob.id)}/segments`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    dirty = false;
    if (data.subtitle_style) setStyle(data.subtitle_style, true);
    showJob(data.job);
    $('result-message').textContent = '修改已保存，字幕文件（含带样式的 .ass）已更新。';
  } catch (error) { $('result-message').textContent = error.message; }
  finally { savePending = false; updateEditorState(); }
}
function hasPartial(job) { return job.status !== 'done' && !!(job.subtitle_count || job.saved_files?.length); }
function canDownload(kind) {
  if (!currentJob) return false;
  if (kind === 'srt') return !!currentJob.has_timestamps;
  if (kind === 'original') return !!currentJob.has_original;
  if (kind === 'bilingual') return !!currentJob.has_timestamps && !!currentJob.has_bilingual;
  if (kind === 'txt') return !!currentJob.has_timestamps;
  return currentJob.status !== 'running' || !!currentJob.has_timestamps;
}
function showJob(job) {
  if (!job?.id) throw new Error('服务没有返回有效任务，需刷新确认是否已提交，避免重复计费。');
  // A delayed cancel response must not roll a completed poll snapshot back to running.
  if (currentJob?.id === job.id && currentJob.status !== 'running' && job.status === 'running') return;
  if (currentJob?.id !== job.id) { cancelRequested = false; $('result-message').textContent = ''; }
  clearTimeout(pollTimer);
  currentJob = job;
  if (job.revision != null) config.submission_revision = job.revision;
  busy = job.status === 'running';
  cancelRequested = cancelRequested || !!job.cancel_requested;
  $('progress-area').hidden = false;
  const progress = Number(job.progress);
  const percent = Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : job.status === 'done' ? 100 : 0;
  $('progress').value = percent;
  $('progress-value').textContent = `${Math.round(percent)}%`;
  $('stage').textContent = job.stage || ({ running: '正在处理', done: '已完成', failed: '处理失败', cancelled: '已取消', empty: '未检测到有效语音' }[job.status] || job.status);
  $('progress-detail').textContent = `语音 ${job.processed_segments ?? 0} / ${job.total_segments ?? '待分析'} 句 · 缓存 ${job.cached_segments ?? 0} 句 · 耗时 ${job.elapsed_seconds ?? 0} 秒`;
  const stateName = { running: '进行中', done: '已完成', failed: '失败', cancelled: '已取消', empty: '无有效语音' }[job.status] || job.status;
  $('result-meta').textContent = `${job.filename || '视频'} · ${job.source_language_name ? `${job.source_language_name} → ` : ''}${job.target_language_name || job.target_language || '原文'} · ${stateName} · ${job.subtitle_count ?? 0} 条字幕`;
  renderEditor(job);
  $('partial-note').hidden = !hasPartial(job);
  $('partial-note').textContent = busy ? '任务尚未完成：当前显示的是部分结果，下载也不是完整字幕。' : '这是未完成任务的部分结果，不是完整字幕；可下载已有时间轴的部分 SRT，请勿当作整片结果。';
  $('job-warning').hidden = !job.warning;
  $('job-warning').textContent = job.warning || '';
  for (const kind of ['srt', 'original', 'bilingual', 'txt', 'json']) $(`download-${kind}`).disabled = !canDownload(kind);
  $('open-output').disabled = !job.output_directory && !job.saved_files?.length;
  $('saved-files').replaceChildren();
  for (const file of job.saved_files || []) {
    const item = document.createElement('li');
    item.textContent = `${FILE_KINDS[file.kind] || file.kind || '结果'} · ${file.path || ''}`;
    $('saved-files').append(item);
  }
  $('saved-area').hidden = !job.saved_files?.length;
  const stats = job.stats;
  const recognition = job.local_asr ? `本地 GPU 识别 ${stats?.asr_requests ?? 0} 句` : `云端识别请求 ${stats?.asr_requests ?? 0} 次 · 已上传音频 ${formatDuration(stats?.uploaded_seconds ?? 0)}（非视频）`;
  const translation = job.translation_model_name ? ` · 翻译：${job.translation_model_name} · 请求 ${stats?.translation_requests ?? 0} 次（缓存 ${stats?.translation_cached ?? 0} 批）· token 输入 ${stats?.prompt_tokens ?? 0} / 输出 ${stats?.completion_tokens ?? 0}${stats?.untranslated ? ` · 未翻译 ${stats.untranslated} 句` : ''}` : '';
  $('job-stats').textContent = stats ? `识别：${job.asr_engine_name || ''} · ${recognition} · 识别缓存 ${stats.asr_cached ?? 0} 句${translation} · 过滤误识别 ${stats.filtered_segments ?? 0} 句 · 跳过语气声 ${(stats.nonverbal_skipped ?? 0) + (stats.recovered_skipped ?? 0)} 句${stats.recovered_segments ? ` · 复查找回 ${stats.recovered_segments} 句（低声或被噪声盖住，建议试听）` : ''}` : '';
  if (busy) {
    status(`${job.stage || '正在生成字幕'}${cancelRequested ? '\n已请求取消：等待当前请求结束，不要重复提交。' : '\n只会查询当前任务进度，不会自动重复提交。'}`);
    schedulePoll(job.id);
  } else if (job.status === 'done') {
    status(job.saved_files?.length ? '生成完成，结果已自动保存。可预览、下载或打开结果文件夹。' : '生成完成，但未报告已保存文件，请查看警告并手动下载结果。', 'success');
  } else if (job.status === 'cancelled') {
    status(`任务已取消。${hasPartial(job) ? '已保留部分结果，可下载核对。' : '暂无可下载的字幕。'}已发出的云端请求可能仍计费。${job.error ? `\n${job.error}` : ''}`);
  } else if (job.status === 'empty') {
    status(job.error || '没有检测到可生成字幕的有效语音。请检查视频和处理范围。');
  } else {
    status(`${job.error || '生成失败，请检查本机服务和网络。'}${hasPartial(job) ? '\n已保留部分结果，不是完整字幕。' : ''}\n不会自动重试；再次生成前请确认用量与计费。`, 'error');
  }
  updateControls();
}
function schedulePoll(id) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (currentJob?.id === id) showJob(data.job);
    } catch (error) {
      if (currentJob?.id !== id) return;
      status(`暂时无法读取原任务进度：${error.message}\n不要重复提交；将继续查询同一任务，不会重跑。`, 'error');
      if (error.status === 404 || error.status === 403) { requestRefresh(); updateControls(); }
      else schedulePoll(id);
    }
  }, 1500);
}
async function loadConfig() {
  config = await api('/api/config');
  fillSelect('target-language', config.languages, 'zh-CN');
  fillSelect('source-language', config.source_languages, 'ja');
  fillSelect('translation-quality', config.translation_models, config.default_translation || 'pro');
  fillSelect('asr-engine', config.asr_engines, config.default_asr || 'siliconflow',
    (item) => item.available === false ? `${item.label}（${item.reason || '不可用'}）` : item.label);
  fillSelect('style-font', config.subtitle_fonts, config.default_style?.font);
  // Start from the options of the last job.
  const last = config.preferences || {};
  for (const [id, value] of [['asr-engine', last.asr_engine], ['source-language', last.source_language], ['target-language', last.target_language], ['translation-quality', last.translation_quality]]) {
    if ([...$(id).options].some((option) => option.value === value && !option.disabled)) $(id).value = value;
  }
  if (typeof last.bilingual === 'boolean') $('bilingual').checked = last.bilingual;
  if (typeof last.skip_nonverbal === 'boolean') $('skip-nonverbal').checked = last.skip_nonverbal;
  fillSelect('custom-format', config.custom_formats, 'chat');
  $('custom-format').value = customSaved().api_format || 'chat';
  $('custom-base-url').value = customSaved().base_url || '';
  $('custom-model').value = customSaved().model || '';
  if (config.subtitle_style) setStyle(config.subtitle_style, true);
  if (config.last_media) acceptMedia(config.last_media, '已恢复上次选择的视频。');
  else resetPreview();
  if (config.latest_job) showJob(config.latest_job);
  else status('已准备好。选择视频后点击“生成字幕”，将处理整个视频。');
  if (config.latest_demosaic) showDemosaic(config.latest_demosaic);
  if (config.desktop_mode !== true) status('当前后端版本不匹配，已禁用提交。请重启客户端后再试。', 'error');
  updateRange();
}
async function refreshConfig() {
  const latest = await api('/api/config');
  config.keys = latest.keys;
  config.asr_engines = latest.asr_engines;
  config.gpu = latest.gpu;
  updateRange();
}
$('generate').addEventListener('click', async () => {
  if ($('generate').disabled || busy || !config || !media) return;
  if (unsaved() && !window.confirm('字幕编辑器里有未保存的修改，开始新任务后这些修改会丢失。继续？')) return;
  const range = processingRange();
  if (config.submission_revision == null) { requestRefresh(); updateControls(); status('缺少任务版本，请刷新后重试。', 'error'); return; }
  const longOrUnknown = range.total == null || range.total > 600 || (range.effective != null && range.effective > 600);
  const repeat = !!currentJob;
  if (longOrUnknown || repeat) {
    const current = engine();
    const destinations = [current?.provider === 'local' ? '语音在本机 GPU 识别' : `语音片段发往${providerName(current?.provider)}`, translating() ? `字幕文本发往${translationDestination()}翻译` : ''].filter(Boolean).join('；');
    const message = `${repeat ? '再次生成是一个新任务，缺失部分可能再次计费。\n' : ''}视频：${media.name || media.filename || '本机视频'}\n总时长：${formatDuration(range.total)}\n范围：${$('range-summary').textContent}\n识别：${current?.label || ''}\n语言：${$('source-language').selectedOptions[0]?.textContent || '原文'} → ${languageName()}${$('bilingual').checked ? '（同时保存双语）' : ''}\n${translating() ? `翻译：${translationItem()?.name || ''}\n` : ''}${costDescription(range)}\n\n视频不会上传；${destinations}。确认开始？`;
    if (!window.confirm(message)) return;
  }
  const data = new FormData();
  data.set('submission_revision', String(config.submission_revision));
  data.set('media_id', media.id);
  data.set('asr_engine', $('asr-engine').value);
  data.set('target_language', $('target-language').value);
  data.set('source_language', $('source-language').value);
  data.set('translation_quality', $('translation-quality').value);
  data.set('context_hint', $('context-hint').value.trim());
  data.set('skip_nonverbal', String($('skip-nonverbal').checked && $('source-language').value === 'ja'));
  data.set('bilingual', String($('bilingual').checked));
  busy = true;
  cancelRequested = false;
  currentJob = null;
  clearTimeout(pollTimer);
  updateControls();
  status('正在向本机服务提交视频与处理设置（不上传视频）…');
  try { showJob((await api('/api/transcribe', { method: 'POST', body: data })).job); }
  catch (error) {
    busy = false;
    // Even a lost response can mean the paid job was accepted. Never re-enable automatic resubmission.
    requestRefresh();
    status(error.status === 409 ? `${error.message}\n页面任务版本已过期，可能有其他窗口已提交。请刷新同步，不会自动重新提交。` : `${error.message}\n无法保证任务未受理。请先刷新查看原任务和服务商用量，避免重复计费。`, 'error');
    updateControls();
  }
});
$('pick-media').addEventListener('click', () => pickMedia());
for (const id of ['custom-base-url', 'custom-model', 'custom-key']) $(id).addEventListener('input', updateControls);
$('custom-format').addEventListener('change', updateControls);
$('custom-save').addEventListener('click', async () => {
  if ($('custom-save').disabled) return;
  try {
    const data = await customCall('/api/custom-translation', customForm(), '正在保存…');
    showCustom(data);
    customMessage(data.key_cleared ? '已保存。接口换了服务器，原来保存的 Key 已清除；如果新接口需要 Key，请填写后再保存一次。' : '已保存。可以点“测试连接”确认能用。', 'success');
  } catch (error) { customMessage(error.message, 'error'); }
  finally { updateRange(); }
});
$('custom-test').addEventListener('click', async () => {
  if ($('custom-test').disabled) return;
  try {
    const data = await customCall('/api/custom-translation/test', customForm(), '正在发送一句测试字幕（最长等待 60 秒）…');
    customMessage(`测试成功（${data.seconds} 秒${data.used_key ? '' : '，未使用 Key'}）：${data.source} → ${data.translation}${customDirty() ? '\n当前填写的内容还没有保存，请点“保存”。' : ''}`, 'success');
  } catch (error) { customMessage(`测试失败：${error.message}`, 'error'); }
  finally { updateControls(); }
});
$('custom-clear').addEventListener('click', async () => {
  if ($('custom-clear').disabled || !window.confirm('清除自定义 API 的地址、模型名称和本机保存的 Key？')) return;
  try {
    showCustom(await customCall('/api/custom-translation/clear', {}, '正在清除…'));
    customMessage('已清除自定义 API 设置。');
  } catch (error) { customMessage(error.message, 'error'); }
  finally { updateRange(); }
});
$('next-untranslated').addEventListener('click', () => {
  const indexes = untranslatedIndexes();
  if (!indexes.length) return;
  lastUntranslated = indexes.find((index) => index > lastUntranslated) ?? indexes[0];
  const list = $('cue-list'), row = list.children[lastUntranslated];
  list.scrollTop = row.offsetTop - list.offsetTop - list.clientHeight / 3;
  row.querySelector('.cue-text').focus({ preventScroll: true });
});
for (const id of ['target-language', 'source-language', 'translation-quality', 'asr-engine']) $(id).addEventListener('input', updateRange);
for (const id of ['video-preview', 'audio-preview']) {
  $(id).addEventListener('loadedmetadata', () => {
    const duration = $(id).duration;
    if (media && !$(id).hidden && Number.isFinite(duration) && duration > 0 && !media.duration_seconds) {
      media.duration_seconds = duration;
      $('selected-info').textContent = `${media.kind === 'video' ? '视频' : '音频'} · ${formatDuration(duration)} · ${formatSize(media.size_bytes)}`;
      updateRange();
    }
  });
  $(id).addEventListener('error', () => {
    if (!$(id).hidden && $(id).hasAttribute('src')) $('media-message').textContent = '当前视频无法在窗口内预览（可能不支持编码），不影响本机提取音频和生成字幕。';
  });
}
$('cancel-job').addEventListener('click', async () => {
  if (!busy || !currentJob || cancelPending || cancelRequested) return;
  if (!window.confirm('请求停止后续处理？已发出的云端请求不能撤销，可能仍计费；已有字幕将作为部分结果保留。')) return;
  const id = currentJob.id;
  cancelPending = true;
  updateControls();
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
    cancelRequested = true;
    if (data.job && currentJob?.id === id) showJob(data.job);
    else status('已请求取消，正在等待当前请求结束并保存部分结果。');
  } catch (error) { status(`取消请求未确认：${error.message}\n仍将查询原任务；不会自动重复提交取消或生成。`, 'error'); }
  finally { cancelPending = false; updateControls(); }
});
$('refresh-page').addEventListener('click', () => window.location.reload());
$('cue-list').addEventListener('input', (event) => {
  const field = event.target.dataset.field;
  if (!field) return;
  const index = Number(event.target.closest('.cue').dataset.index);
  cues[index][field] = event.target.value;
  renderSubtitle();
  markDirty();
});
$('cue-list').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-act]');
  if (!button) return;
  const index = Number(button.closest('.cue').dataset.index);
  const video = $('editor-video');
  const act = button.dataset.act;
  if (act === 'seek') {
    video.currentTime = cues[index].start + 0.001;
    video.play().catch(() => {});
    return;
  }
  if (act === 'delete') {
    cues.splice(index, 1);
    renderCueList();
    rebuildOverlay();
    markDirty();
    return;
  }
  const now = Math.round(video.currentTime * 1000) / 1000;
  cues[index][act] = now;
  button.closest('.cue').querySelector('.cue-time').textContent = `#${index + 1}\n${cueTime(cues[index].start)}\n→ ${cueTime(cues[index].end)}`;
  refreshOverlayCue(index);
  markDirty();
});
$('editor-video').addEventListener('timeupdate', () => {
  const index = findActiveCue($('editor-video').currentTime);
  if (index === activeIndex) return;
  const list = $('cue-list');
  list.children[activeIndex]?.classList.remove('active');
  list.children[index]?.classList.add('active');
  activeIndex = index;
  // Follow playback, but never move the list while the user is typing in it.
  if (index >= 0 && $('follow-playback').checked && !list.contains(document.activeElement)) {
    const row = list.children[index];
    list.scrollTop = row.offsetTop - list.offsetTop - list.clientHeight / 3;
  }
});
$('editor-video').addEventListener('error', () => {
  if ($('editor-video').hasAttribute('src')) $('result-message').textContent = '找不到或无法播放这个任务的视频，仍可以直接修改字幕文字。';
});
const PLAYERS = ['video-preview', 'editor-video', 'demosaic-video'];
for (const id of PLAYERS) {
  $(id).addEventListener('play', () => { for (const other of PLAYERS) if (other !== id) $(other).pause(); });
}
$('video-preview').addEventListener('timeupdate', updateDemosaicControls);
$('demosaic-sample').addEventListener('click', () => startDemosaic(true));
$('demosaic-full').addEventListener('click', () => startDemosaic(false));
$('demosaic-cancel').addEventListener('click', async () => {
  if (!demosaicRunning() || $('demosaic-cancel').disabled) return;
  if (!window.confirm('停止去马赛克？已处理的部分不会保存。')) return;
  const id = demosaicJob.id;
  demosaicPending = true;
  updateControls();
  try {
    const data = await api(`/api/demosaic/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
    demosaicPending = false;
    if (demosaicJob?.id === id) showDemosaic(data.job);
  } catch (error) { demosaicMessage(`取消请求未确认：${error.message}`, 'error'); }
  finally { demosaicPending = false; updateControls(); }
});
$('demosaic-reveal').addEventListener('click', async () => {
  if (!demosaicJob) return;
  try { await api(`/api/demosaic/${encodeURIComponent(demosaicJob.id)}/reveal`, { method: 'POST' }); }
  catch (error) { demosaicMessage(error.message, 'error'); }
});
$('demosaic-video').addEventListener('error', () => {
  if ($('demosaic-video').hasAttribute('src')) demosaicMessage('窗口内无法预览这个视频（H.265 需要显卡硬解或 HEVC 扩展），请点“在文件夹中显示”用播放器打开。');
});
$('editor-video').addEventListener('loadedmetadata', layoutSubtitles);
$('editor-video').addEventListener('dblclick', (event) => { event.preventDefault(); toggleFullscreen(); });
// The native control bar covers the bottom of the picture: show it only while the pointer is over the video,
// so the subtitle position can be judged while adjusting the style.
$('video-stage').addEventListener('pointerenter', () => { $('editor-video').controls = true; });
$('video-stage').addEventListener('pointerleave', () => { if (!document.fullscreenElement) $('editor-video').controls = false; });
new ResizeObserver(layoutSubtitles).observe($('editor-video'));
document.addEventListener('fullscreenchange', () => {
  $('fullscreen-preview').textContent = document.fullscreenElement ? '退出全屏' : '全屏预览';
  layoutSubtitles();
});
$('fullscreen-preview').addEventListener('click', toggleFullscreen);
$('overlay-mode').addEventListener('change', renderSubtitle);
for (const id of ['style-font', 'style-size', 'style-color', 'style-border', 'style-outline-color', 'style-position', 'style-margin']) {
  $(id).addEventListener('input', () => { subtitleStyle = readStyle(); applyStyle(); updateEditorState(); });
}
$('style-reset').addEventListener('click', () => {
  if (!config?.default_style) return;
  setStyle(config.default_style);
  updateEditorState();
});
$('save-edits').addEventListener('click', saveEdits);
$('revert-edits').addEventListener('click', () => {
  if (!currentJob || !unsaved()) return;
  loadEditorCues(currentJob);
  if (savedStyle) setStyle(savedStyle);
  updateEditorState();
  $('result-message').textContent = '已撤销未保存的修改。';
});
$('restore-generated').addEventListener('click', async () => {
  if (!currentJob || !window.confirm('把字幕恢复为最初生成的版本？你保存过的修改会被覆盖。')) return;
  savePending = true;
  updateEditorState();
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(currentJob.id)}/segments/restore`, { method: 'POST' });
    dirty = false;
    if (data.subtitle_style) savedStyle = { ...data.subtitle_style };
    showJob(data.job);
    $('result-message').textContent = '已恢复为生成版本，字幕文件已更新。';
  } catch (error) { $('result-message').textContent = error.message; }
  finally { savePending = false; updateEditorState(); }
});
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's' && !$('editor').hidden) {
    event.preventDefault();
    saveEdits();
  }
});
window.addEventListener('beforeunload', (event) => {
  if (unsaved()) { event.preventDefault(); event.returnValue = ''; }
});
for (const provider of PROVIDER_IDS) {
  $(`save-key-${provider}`).addEventListener('click', async () => {
    if ($(`save-key-${provider}`).disabled) return;
    const field = $(`api-key-${provider}`);
    if (!field.value.trim()) { $('key-message').textContent = `请输入${providerName(provider)}的 API Key。`; return; }
    keyPending = true;
    updateControls();
    try {
      await api('/api/key', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider, api_key: field.value.trim() }) });
      $('key-message').textContent = `${providerName(provider)} Key 已安全保存到本机；页面不会显示已保存的 Key。`;
      await refreshConfig();
    } catch (error) { $('key-message').textContent = error.message; }
    finally { field.value = ''; keyPending = false; updateControls(); }
  });
  $(`clear-key-${provider}`).addEventListener('click', async () => {
    if ($(`clear-key-${provider}`).disabled || !window.confirm(`清除本机保存的${providerName(provider)} Key？不会删除云端账户或云端 Key。`)) return;
    keyPending = true;
    updateControls();
    try {
      await api('/api/key/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider }) });
      $(`api-key-${provider}`).value = '';
      $('key-message').textContent = `已清除本机保存的${providerName(provider)} Key。`;
      await refreshConfig();
    } catch (error) { $('key-message').textContent = error.message; }
    finally { keyPending = false; updateControls(); }
  });
}
for (const kind of ['srt', 'original', 'bilingual', 'txt', 'json']) {
  $(`download-${kind}`).addEventListener('click', async () => {
    if (!canDownload(kind)) return;
    const job = currentJob;
    const button = $(`download-${kind}`);
    button.disabled = true;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(job.id)}/download/${kind}`, { credentials: 'same-origin', cache: 'no-store', mode: 'same-origin' });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `下载失败：HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      // Keep a backend-provided filename when available, with a safe fallback for partial results.
      const disposition = response.headers.get('Content-Disposition') || '';
      const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i);
      const plain = disposition.match(/filename="([^"]+)"/i);
      let filename = '';
      try { filename = encoded ? decodeURIComponent(encoded[1]) : plain?.[1] || ''; } catch { /* use fallback */ }
      link.download = (filename || `${job.filename || 'subtitle'}${job.status === 'done' ? '' : '.partial'}.${kind === 'txt' || kind === 'json' ? kind : `${kind}.srt`}`).replace(/[\\/:*?"<>|]/g, '_');
      link.href = url;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      $('result-message').textContent = job.status === 'done' ? '已开始下载。' : '已开始下载部分结果；任务未完成，不是完整字幕。';
    } catch (error) { $('result-message').textContent = error.message; }
    finally { button.disabled = !canDownload(kind); }
  });
}
$('open-output').addEventListener('click', async () => {
  if (!currentJob || $('open-output').disabled) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(currentJob.id)}/open-output`, { method: 'POST' });
    $('result-message').textContent = '已请求在 Windows 中打开结果文件夹。';
  } catch (error) { $('result-message').textContent = error.message; }
});
loadConfig().catch((error) => {
  config = null;
  requestRefresh();
  updateControls();
  status(`无法读取本机配置：${error.message}\n请确认服务已启动并刷新；不会自动提交任务。`, 'error');
});

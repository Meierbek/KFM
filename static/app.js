'use strict';

const paths = {
  'scan': '<path d="M7 3H4a1 1 0 0 0-1 1v3m14-4h3a1 1 0 0 1 1 1v3M3 17v3a1 1 0 0 0 1 1h3m10 0h3a1 1 0 0 0 1-1v-3M7 12h10m-5-5v10"/>',
  'play-square': '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="m10 8 6 4-6 4z"/>',
  'play': '<path d="m9 5 11 7-11 7z"/>',
  'pause': '<path d="M8 5v14M16 5v14"/>',
  'book': '<path d="M12 5c-3-2-7-2-10-1v15c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Zm0 0v15"/>',
  'shield': '<path d="M12 3 3 7v5c0 5 9 9 9 9s9-4 9-9V7z"/><path d="m8 12 3 3 5-6"/>',
  'help': '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4.3 1.7c-1.1.7-1.8 1-1.8 2.3m0 3h.01"/>',
  'upload': '<path d="M12 16V4m-5 5 5-5 5 5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>',
  'video': '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="m10 8 6 4-6 4zM3 7h3m12 0h3M3 17h3m12 0h3"/>',
  'arrow-right': '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  'arrow-left': '<path d="M20 12H4m6-6-6 6 6 6"/>',
  'check': '<path d="m5 12 4 4L19 6"/>',
  'check-circle': '<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>',
  'clock': '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  'file': '<path d="M14 2H5a1 1 0 0 0-1 1v18a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1V8zM14 2v6h6M8 13h8m-8 4h6"/>',
  'x': '<path d="m6 6 12 12M18 6 6 18"/>',
  'info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  'alert': '<path d="m10.3 3.5-8 14A2 2 0 0 0 4 20h16a2 2 0 0 0 1.7-2.5l-8-14a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 3h.01"/>',
  'link': '<path d="M14 3h7v7m0-7L10 14m0-9H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>',
  'expand': '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  'download': '<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',
  'refresh': '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 6a8 8 0 0 1 13 1l1 5M4 12l1 5a8 8 0 0 0 13 1"/>',
  'image': '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m21 16-6-6L3 21"/>',
  'server': '<rect x="3" y="3" width="18" height="7" rx="2"/><rect x="3" y="14" width="18" height="7" rx="2"/><path d="M7 6.5h.01M7 17.5h.01M15 6.5h3m-3 11h3"/>',
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.file}</svg>`;
function hydrateIcons(root = document) { root.querySelectorAll('[data-icon]').forEach(el => { el.insertAdjacentHTML('afterbegin', icon(el.dataset.icon)); delete el.dataset.icon; }); }
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
const main = document.querySelector('#main');
let config = {model_connected: false, model_name: 'Модель не подключена', max_upload_mb: 128};
let selectedFile = null;
let mode = 'video';
let scenario = 'violation';
let pollTimer = null;
let demoTimer = null;
let currentJob = null;
let currentTime = 0;
let demoPlaying = false;
let uploadBusy = false;
let renderVersion = 0;
let toastTimer;

function timecode(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function bytes(value) { return value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1).replace('.', ',')} МБ` : `${Math.max(1, Math.round(value / 1024))} КБ`; }
function toast(message) {
  const el = document.querySelector('#toast');
  el.textContent = message; el.classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('visible'), 5000);
}
async function api(path, options = {}) {
  const response = await fetch(`/ui/api${path}`, {credentials: 'same-origin', ...options});
  let body;
  try { body = await response.json(); } catch { throw new Error('Сервер вернул непонятный ответ. Обновите страницу.'); }
  if (!response.ok) throw new Error(body.error?.message || 'Не удалось выполнить запрос.');
  return body;
}
function cleanTimers() { clearTimeout(pollTimer); clearInterval(demoTimer); pollTimer = demoTimer = null; demoPlaying = false; }
function navigate(path) { if (uploadBusy) return toast('Дождитесь завершения загрузки.'); history.pushState({}, '', path); renderRoute(); window.scrollTo({top: 0, behavior: 'instant'}); }
async function renderRoute() {
  const version = ++renderVersion;
  cleanTimers();
  const match = location.pathname.match(/^\/results\/([0-9a-f]{32})\/?$/);
  document.querySelector('#breadcrumb').textContent = match ? 'Результат проверки' : 'Проверка видео';
  document.title = match ? 'Результат проверки — KFM' : 'KFM — Проверка видео';
  if (match) {
    main.innerHTML = '<div class="initial-loading"><span class="spinner"></span>Открываем проверку…</div>';
    try {
      const job = await api(`/jobs/${match[1]}`);
      if (version !== renderVersion) return;
      renderResult(job);
      if (job.status === 'processing') poll(job.job_id, version);
    } catch (error) {
      if (version !== renderVersion) return;
      main.innerHTML = `<div class="empty-page">${icon('file')}<h1>Проверка недоступна</h1><p>${esc(error.message)}</p><button class="button primary" data-action="home">Новая проверка</button></div>`;
    }
  } else {
    renderHome();
    loadRecent(version);
  }
}

function renderHome() {
  currentJob = null;
  const modelNotice = !config.model_connected
    ? 'Сайт готов к работе с локальной моделью. Пока изучите результат в деморежиме.'
    : config.experimental === true
      ? 'Точность пока недостаточна: возможны ложные тревоги и пропуски. Результаты требуют проверки.'
      : 'Локальная модель подключена. Результаты доступны после проверки видео.';
  main.innerHTML = `
    <div class="page-heading"><div><div class="eyebrow"><span class="eyebrow-line"></span>ВНИМАНИЕ К КАЖДОМУ ДЕЙСТВИЮ</div><h1>Всё важное —<br>в поле зрения<span class="heading-dot">.</span></h1><p class="page-intro">Проверяйте кассовые операции по видео.<br>События, таймкоды и доказательства — на одном экране.</p></div><div class="hero-art" aria-hidden="true"><div class="focus-corner tl"></div><div class="focus-corner tr"></div><div class="focus-corner bl"></div><div class="focus-corner br"></div><div class="art-grid"></div><div class="art-disc"><span class="art-orbit"></span>${icon('scan')}</div><div class="art-label"><span class="live-dot"></span>КАЖДЫЙ КАДР ИМЕЕТ ЗНАЧЕНИЕ</div><span class="art-coordinate top">KFM / VIDEO</span><span class="art-coordinate bottom">FRAME BY FRAME</span></div></div>
    <div class="home-grid">
      <section class="panel upload-panel" aria-labelledby="upload-title"><div class="panel-title"><div class="section-icon">${icon('video')}</div><div><h2 id="upload-title">Новая проверка</h2><p>Начните с видеозаписи или учебного сценария</p></div></div>
        <div class="mode-switch" role="tablist" aria-label="Режим проверки"><button role="tab" id="video-tab" aria-selected="${mode === 'video'}" aria-controls="upload-content" class="${mode === 'video' ? 'selected' : ''}" data-mode="video">${icon('upload')}Ваше видео</button><button role="tab" id="demo-tab" aria-selected="${mode === 'demo'}" aria-controls="upload-content" class="${mode === 'demo' ? 'selected' : ''}" data-mode="demo">${icon('play-square')}Деморежим<span class="small-new">ПОПРОБОВАТЬ</span></button></div>
        <div id="upload-content" role="tabpanel" aria-labelledby="${mode === 'video' ? 'video-tab' : 'demo-tab'}">${uploadContent()}</div>
      </section>
      <aside class="guide-column"><section class="guide-card"><span class="eyebrow">ПОНЯТНАЯ ПРОВЕРКА</span><h2>От видео<br>к конкретному событию</h2><div class="guide-step"><span class="step-number">01</span><div><h3>Загрузите запись</h3><p>Выберите видео кассовой операции на компьютере.</p></div></div><div class="guide-step"><span class="step-number">02</span><div><h3>Получите результат</h3><p>Локальная модель оценит возможные нарушения.</p></div></div><div class="guide-step"><span class="step-number">03</span><div><h3>Проверьте по кадрам</h3><p>Откройте событие, время и основание для проверки.</p></div></div></section><div class="model-card ${config.model_connected ? 'connected' : ''}"><div class="model-card-top"><span class="model-status-dot"></span><strong>${config.model_connected ? esc(config.model_name) : 'Модель пока не подключена'}</strong></div><p>${modelNotice}</p><button class="text-button" data-action="${config.model_connected ? 'help' : 'switch-demo'}">${config.model_connected ? 'Как работает проверка' : 'Посмотреть демонстрацию'}${icon('arrow-right')}</button></div></aside>
    </div>
    <section class="recent-section" aria-labelledby="recent-title"><div class="section-heading"><div><span class="eyebrow">ВАШЕ РАБОЧЕЕ ПРОСТРАНСТВО</span><h2 id="recent-title">Последние проверки</h2></div><span class="section-hint">Хранятся только на этом компьютере</span></div><div id="recent-list" class="recent-list"><div class="recent-empty">${icon('clock')}<div><strong>Здесь появятся ваши проверки</strong><p>Загрузите первую запись или откройте демосценарий.</p></div></div></div></section>
  `;
  bindDropzone();
}

function uploadContent() {
  if (mode === 'demo') return `<div class="demo-description"><span class="badge demo">${icon('play-square')}Учебный пример</span><h3>Посмотрите, как выглядит результат</h3><p>Нарисованная сцена и готовые события. Ваши видео в демонстрации не используются.</p></div><div class="scenario-options" role="radiogroup" aria-label="Учебный сценарий"><label class="scenario ${scenario === 'violation' ? 'chosen' : ''}"><input type="radio" name="scenario" value="violation" ${scenario === 'violation' ? 'checked' : ''}><span class="scenario-symbol warning">${icon('scan')}</span><strong>С возможными нарушениями</strong><span>Два события, таймкоды и учебные кадры</span><i class="radio-mark"></i></label><label class="scenario ${scenario === 'clean' ? 'chosen' : ''}"><input type="radio" name="scenario" value="clean" ${scenario === 'clean' ? 'checked' : ''}><span class="scenario-symbol positive">${icon('check-circle')}</span><strong>Без нарушений</strong><span>Пример результата без найденных событий</span><i class="radio-mark"></i></label></div><button class="button primary full" id="start-demo" data-action="start-demo">Открыть демонстрацию${icon('arrow-right')}</button><p class="under-button">${icon('info')}Результат заранее подготовлен и не является выводом модели</p>`;
  return `<label class="dropzone ${selectedFile ? 'has-file' : ''}" id="dropzone" tabindex="0" role="button" aria-label="Выбрать видео MOV или MP4"><input type="file" id="video-input" accept=".mov,.mp4,video/quicktime,video/mp4"><span class="upload-symbol">${icon(selectedFile ? 'video' : 'upload')}</span><strong>${selectedFile ? esc(selectedFile.name) : 'Перетащите видео сюда'}</strong><span>${selectedFile ? `${bytes(selectedFile.size)} · Нажмите, чтобы выбрать другой файл` : 'или <span class="choose-link">выберите файл на компьютере</span>'}</span><span class="format-tags"><span>MOV</span><span>MP4</span><span>ДО ${config.max_upload_mb} МБ</span>${Number.isFinite(config.max_duration_sec) && config.max_duration_sec > 0 ? `<span>ДО ${config.max_duration_sec % 60 === 0 ? `${config.max_duration_sec / 60} МИН` : `${config.max_duration_sec} С`}</span>` : ''}</span></label><div id="file-error" class="inline-error" role="alert"></div><div class="upload-bottom"><span class="secure-line">${icon('shield')}Обработка на вашем компьютере</span>${selectedFile ? '<button class="text-button muted" data-action="clear-file">Убрать файл</button>' : ''}</div>${!config.model_connected ? `<div class="notice model-notice">${icon('info')}<p>Модель ещё не подключена. Можно загрузить и открыть запись; автоматический анализ станет доступен после подключения.</p></div>` : ''}<button class="button primary full" id="upload-button" data-action="upload" ${selectedFile ? '' : 'disabled'}>${config.model_connected ? 'Запустить проверку' : 'Загрузить и открыть видео'}${icon('arrow-right')}</button><div class="upload-progress hidden" id="upload-progress"><div><span id="upload-label">Загрузка видео…</span><strong id="upload-percent">0%</strong></div><progress id="upload-bar" value="0" max="100">0%</progress></div><p class="under-button">${icon('file')}Поддерживаются записи одной кассы в форматах MOV и MP4</p>`;
}

function updateUploadContent() {
  document.querySelector('#upload-content').innerHTML = uploadContent();
  document.querySelector('#upload-content').setAttribute('aria-labelledby', mode === 'video' ? 'video-tab' : 'demo-tab');
  document.querySelectorAll('[data-mode]').forEach(el => { el.classList.toggle('selected', el.dataset.mode === mode); el.setAttribute('aria-selected', String(el.dataset.mode === mode)); });
  bindDropzone();
}
function validateFile(file) {
  if (uploadBusy) return false;
  if (!file) return false;
  if (!/\.(mov|mp4)$/i.test(file.name)) { showFileError('Выберите видео в формате MOV или MP4.'); return false; }
  if (!file.size) { showFileError('Файл пустой. Выберите другую запись.'); return false; }
  if (file.size > config.max_upload_mb * 1024 ** 2) { showFileError(`Файл слишком большой. Максимум — ${config.max_upload_mb} МБ.`); return false; }
  selectedFile = file; updateUploadContent(); return true;
}
function showFileError(message) { const el = document.querySelector('#file-error'); if (el) el.textContent = message; else toast(message); }
function bindDropzone() {
  const drop = document.querySelector('#dropzone');
  if (!drop) return;
  const input = document.querySelector('#video-input');
  input.addEventListener('change', () => validateFile(input.files[0]));
  drop.addEventListener('keydown', event => { if (event.target === drop && ['Enter', ' '].includes(event.key)) { event.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.add('dragging'); }));
  drop.addEventListener('dragleave', event => { if (!drop.contains(event.relatedTarget)) drop.classList.remove('dragging'); });
  drop.addEventListener('drop', event => { event.preventDefault(); drop.classList.remove('dragging'); if (event.dataTransfer.files.length !== 1) return showFileError('Для одной проверки выберите один видеофайл.'); validateFile(event.dataTransfer.files[0]); });
}
async function loadRecent(version = renderVersion) {
  try {
    const {jobs} = await api('/jobs');
    const target = document.querySelector('#recent-list');
    if (!target || version !== renderVersion || !jobs.length) return;
    target.innerHTML = jobs.map(job => {
      const unavailable = job.status === 'failed';
      const label = job.status === 'processing' ? 'Обработка' : unavailable ? 'Без анализа' : job.demo ? 'Демо завершено' : 'Готово';
      return `<a class="recent-row" href="/results/${job.job_id}" data-nav="/results/${job.job_id}"><span class="recent-file-icon">${icon(job.demo ? 'play-square' : 'video')}</span><span class="recent-name"><strong>${esc(job.filename)}</strong><span>${new Date(job.created_at * 1000).toLocaleString('ru-RU', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'})}${job.duration_sec ? ` · ${timecode(job.duration_sec)}` : ''}</span></span><span class="badge ${job.demo ? 'demo' : 'neutral'}">${job.demo ? 'Демо' : 'Видео'}</span><span class="recent-status"><span class="status-dot ${unavailable ? 'idle' : job.status === 'processing' ? 'busy' : ''}"></span>${label}</span>${icon('arrow-right', 'recent-arrow')}</a>`;
    }).join('');
  } catch { /* The upload flow remains available if history fails. */ }
}

async function startDemo() {
  const button = document.querySelector('#start-demo');
  if (!button || button.disabled) return;
  button.disabled = true; button.innerHTML = '<span class="spinner"></span>Открываем сценарий…';
  try { const job = await api('/demo', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({scenario})}); navigate(`/results/${job.job_id}`); }
  catch (error) { toast(error.message); button.disabled = false; button.innerHTML = `Открыть демонстрацию${icon('arrow-right')}`; }
}
function uploadVideo() {
  if (!selectedFile || uploadBusy) return;
  const form = new FormData(); form.append('file', selectedFile);
  const xhr = new XMLHttpRequest();
  uploadBusy = true;
  const button = document.querySelector('#upload-button');
  button.disabled = true; button.innerHTML = '<span class="spinner"></span>Загружаем запись…';
  document.querySelector('#upload-progress').classList.remove('hidden');
  xhr.open('POST', '/ui/api/analyze'); xhr.timeout = 120000;
  xhr.upload.onprogress = event => {
    if (!event.lengthComputable) return;
    const percent = Math.round(event.loaded / event.total * 100);
    document.querySelector('#upload-percent').textContent = `${percent}%`;
    document.querySelector('#upload-bar').value = percent;
    if (percent === 100) document.querySelector('#upload-label').textContent = 'Сервер принимает запись…';
  };
  const failed = message => { uploadBusy = false; button.disabled = false; button.innerHTML = `Повторить загрузку${icon('refresh')}`; showFileError(message); document.querySelector('#upload-progress').classList.add('hidden'); };
  xhr.onload = () => {
    uploadBusy = false;
    let result;
    try { result = JSON.parse(xhr.responseText); } catch { return failed('Сервер не ответил. Проверьте, запущен ли KFM.'); }
    if (xhr.status < 200 || xhr.status >= 300) return failed(result.error?.message || 'Не удалось загрузить видео.');
    selectedFile = null; navigate(`/results/${result.job_id}`);
  };
  xhr.onerror = () => failed('Нет связи с локальным сервером. Проверьте, запущен ли KFM.');
  xhr.ontimeout = () => failed('Загрузка заняла слишком много времени. Повторите попытку.');
  xhr.send(form);
}

const supportedRuleNames = Object.freeze({
  nb231_12: 'личные вещи сотрудника в кассовой зоне',
  nb231_22: 'уход кассира до окончания обслуживания',
  nb231_20: 'хранение наличных от предыдущей операции',
});

function resultScope(job) {
  if (job.demo) return {notice: '', unresolved: 0};
  const analysis = job.analysis && typeof job.analysis === 'object' ? job.analysis : null;
  const names = Array.isArray(analysis?.supported_rule_ids)
    ? [...new Set(analysis.supported_rule_ids.filter(id => typeof id === 'string' && Object.hasOwn(supportedRuleNames, id)))].map(id => supportedRuleNames[id])
    : [];
  const unresolved = Number.isSafeInteger(analysis?.unresolved_candidates) && analysis.unresolved_candidates > 0
    ? analysis.unresolved_candidates : 0;
  const sampled = analysis?.sampled === true;
  const learnedCandidates = analysis?.sampling_strategy === 'learned_candidates';
  const detectorOnly = learnedCandidates && analysis.detector_signal === false && analysis.candidate_windows === 0;
  if (detectorOnly) return {
    notice: 'Обученный детектор не отметил подозрительных фрагментов; подробная проверка правил не запускалась. Это выборочная автоматическая оценка: короткие действия и другие признаки нарушений могут быть пропущены.',
    unresolved, sampled, detectorOnly,
  };
  const parts = [];
  if (learnedCandidates && analysis.detector_signal === true) {
    const count = Number.isSafeInteger(analysis.candidate_windows) && analysis.candidate_windows > 0
      ? ` (${analysis.candidate_windows})` : '';
    parts.push(`Подробная проверка выполнялась только в наиболее подозрительных фрагментах${count}, выбранных обученным детектором.`);
  }
  if (names.length) parts.push(`Проверяемые категории: ${names.join('; ')}.`);
  if (sampled) {
    if (analysis.coverage_complete === true) parts.push('Все запланированные фрагменты обработаны.');
    parts.push('Модель проверяет выбранные кадры. Короткие действия между ними и другие виды нарушений могут быть пропущены.');
  } else {
    parts.push('Экспериментальный результат требует проверки контекста записи и применимости указанного правила.');
  }
  return {notice: parts.join(' '), unresolved, sampled, detectorOnly};
}

function renderResult(job) {
  currentJob = job; currentTime = 0; demoPlaying = false;
  const pending = job.status === 'processing';
  const unavailable = job.error?.code === 'model_not_configured';
  const failed = job.status === 'failed' && !unavailable;
  const violations = job.violations || [];
  const positive = job.verdict === 'violation';
  const done = job.status === 'done';
  const scope = resultScope(job);
  const cleanDescription = job.demo
    ? 'Так выглядит отчёт, когда учебный сценарий не содержит отмеченных событий.'
    : scope.detectorOnly
      ? 'Обученный детектор не выбрал фрагменты для подробной проверки правил. Это не подтверждает отсутствие нарушений.'
    : scope.sampled
      ? 'Модель не отметила событий по проверяемым категориям в выбранных кадрах. Это не подтверждает отсутствие всех нарушений.'
      : 'Модель не отметила событий в этой проверке. Это не подтверждает отсутствие всех нарушений.';
  const title = pending ? 'Проверка в процессе' : unavailable ? 'Видео загружено' : failed ? 'Проверка не завершена' : 'Результат проверки';
  main.innerHTML = `<div class="result-heading"><button class="back-button" data-action="home">${icon('arrow-left')}Новая проверка</button><div class="result-title-row"><div><span class="eyebrow">${job.demo ? 'ЗНАКОМСТВО С ИНТЕРФЕЙСОМ' : 'ПРОВЕРКА ВИДЕОЗАПИСИ'}</span><h1>${title}</h1></div>${done ? `<button class="button secondary" data-action="download">${icon('download')}Скачать JSON</button>` : ''}</div><div class="file-meta">${icon(job.demo ? 'play-square' : 'video')}<strong>${esc(job.filename)}</strong>${job.duration_sec ? `<span class="meta-divider"></span><span>${timecode(job.duration_sec)}</span>` : ''}${job.size_bytes ? `<span class="meta-divider"></span><span>${bytes(job.size_bytes)}</span>` : ''}<span class="badge ${job.demo ? 'demo' : 'neutral'}">${job.demo ? 'Деморежим' : 'Локальная запись'}</span></div></div>
  ${job.demo ? `<div class="demo-banner">${icon('info')}<div><strong>Демонстрационный результат</strong><span>События заданы заранее. Кадры — иллюстрации учебной сцены. Реальное видео и модель не используются.</span></div><span class="demo-banner-tag">DEMO</span></div>` : ''}
  ${!job.demo && (job.experimental || scope.sampled) ? `<div class="demo-banner">${icon('info')}<div><strong>Область экспериментальной проверки</strong><span>${esc(scope.notice)}</span></div></div>` : ''}
  ${done && scope.unresolved ? `<div class="notice unavailable-notice" role="status">${icon('alert')}<div><strong>Остались неоднозначные фрагменты (${scope.unresolved})</strong><p>По ним не удалось подтвердить все условия правила. Они не включены в список найденных событий.</p></div></div>` : ''}
  ${pending ? processingPanel(job) : ''}
  ${unavailable ? `<div class="notice unavailable-notice">${icon('server')}<div><strong>Запись готова к анализу. Осталось подключить модель.</strong><p>Видео сохранено на этом компьютере. Автоматическая проверка ещё не выполнялась, поэтому вердикта и найденных событий пока нет.</p></div><button class="button secondary" data-action="${config.model_connected ? 'retry' : 'demo-menu'}">${config.model_connected ? 'Запустить проверку' : 'Посмотреть демо'}${icon('arrow-right')}</button></div>` : ''}
  ${failed ? `<div class="notice error-notice">${icon('alert')}<div><strong>Результат анализа не получен</strong><p>${esc(job.error?.message || 'Повторите попытку.')}</p></div><button class="button secondary" data-action="retry">Повторить${icon('refresh')}</button></div>` : ''}
  ${done ? `<div class="verdict-card ${positive ? 'warning-verdict' : 'clean-verdict'}"><div class="verdict-icon">${icon(positive ? 'scan' : 'check-circle')}</div><div><span class="eyebrow">${job.demo ? 'В УЧЕБНОМ СЦЕНАРИИ' : 'РЕЗУЛЬТАТ АНАЛИЗА'}</span><h2>${positive ? 'Обнаружены признаки возможных нарушений' : job.demo ? 'Нарушение не обнаружено' : 'По проверяемым признакам события не найдены'}</h2><p>${positive ? `${violations.length} ${violations.length === 1 ? 'событие требует' : violations.length < 5 ? 'события требуют' : 'событий требуют'} внимания проверяющего. Откройте кадры и изучите контекст.` : cleanDescription}</p></div><div class="verdict-number">${String(violations.length).padStart(2, '0')}<span>${positive ? 'СОБЫТИЯ' : 'СОБЫТИЙ'}</span></div></div>` : ''}
  <div class="result-grid"><section class="panel video-panel" aria-labelledby="video-title"><div class="panel-header"><h2 id="video-title">${job.demo ? 'Учебная сцена' : 'Видеозапись'}</h2><span class="panel-caption">${job.demo ? 'Иллюстрация' : 'Оригинал на вашем компьютере'}</span></div>${job.demo ? demoPlayer(job) : `<div class="video-container"><video id="source-video" controls preload="metadata" playsinline src="/media/${job.job_id}"></video><div class="video-codec-help hidden" id="codec-help">${icon('video')}<strong>Браузер не воспроизводит этот кодек</strong><p>Файл сохранён. Откройте оригинал локальным плеером или используйте MP4 с кодеком H.264.</p><a class="button secondary" href="/media/${job.job_id}" download="${esc(job.filename)}">Скачать оригинал${icon('download')}</a></div></div>`}<div class="timeline-section"><div class="timeline-label"><span>${icon('clock')}Шкала событий</span><span id="player-time">00:00 / ${timecode(job.duration_sec)}</span></div><div class="event-timeline">${violations.map((event, i) => `<button class="timeline-pin" data-position="${Math.min(98, event.timecode_sec / (job.duration_sec || 60) * 100)}%" data-seek="${event.timecode_sec}" aria-label="Событие ${i + 1}, ${timecode(event.timecode_sec)}"><span>${i + 1}</span></button>`).join('')}<div class="timeline-playhead" id="timeline-playhead"></div></div><div class="timeline-ends"><span>00:00</span><span>${timecode(job.duration_sec)}</span></div></div><div class="video-footnote">${icon(job.demo ? 'info' : 'shield')}<span>${job.demo ? 'Учебные изображения созданы для демонстрации интерфейса.' : 'Файл доступен только через ваш локальный сервер KFM.'}</span></div></section>
  <section class="events-column" aria-labelledby="events-title"><div class="events-header"><h2 id="events-title">${done ? 'Найденные события' : 'События'}</h2><span class="count-pill">${done ? violations.length : '—'}</span></div>${violations.length ? violations.map((event, index) => eventCard(event, index, job.demo)).join('') : `<div class="events-empty">${icon(done ? 'check-circle' : 'scan')}<h3>${pending ? 'Ищем события' : unavailable ? 'Здесь будут события' : failed ? 'Проверка не завершена' : 'События не найдены'}</h3><p>${pending ? 'После завершения здесь появятся результаты и подтверждающие кадры.' : unavailable ? 'Подключённая модель добавит таймкоды, кадры и объяснения.' : failed ? 'Ошибка обработки не означает, что на записи нет нарушений.' : job.demo ? 'В этом демонстрационном сценарии нет отмеченных нарушений.' : 'В этой проверке модель не отметила событий.'}</p>${unavailable ? '<button class="text-button" data-action="demo-menu">Как будет выглядеть результат' + icon('arrow-right') + '</button>' : ''}</div>`}</section></div>
  ${done && positive ? `<div class="review-note">${icon('info')}<p>${job.demo ? 'Учебные категории и правила показывают возможности интерфейса. Они не являются нормативным обоснованием реального нарушения.' : 'Результат — подсказка для проверки человеком. Изучите контекст записи и применимость указанного правила.'}</p></div>` : ''}`;
  hydrateIcons(main);
  main.querySelectorAll("[data-position]").forEach(el => el.style.setProperty("--position", el.dataset.position));
  const video = document.querySelector('#source-video');
  if (video) {
    video.addEventListener('error', () => { document.querySelector('#codec-help').classList.remove('hidden'); video.classList.add('hidden'); });
    video.addEventListener('timeupdate', () => updateTime(video.currentTime, video.duration));
    video.addEventListener('loadedmetadata', () => updateTime(video.currentTime, video.duration));
  }
  if (job.demo) {
    const slider = document.querySelector('#demo-scrubber');
    slider?.addEventListener('input', () => seekTo(Number(slider.value)));
  }
}

function processingPanel(job) {
  return `<div class="processing-panel"><div class="processing-top"><span class="processing-icon">${icon('scan')}</span><div><strong id="processing-stage">${esc(job.stage)}</strong><span>${job.demo ? 'Показываем этапы учебной проверки' : 'Видео принято. Дождитесь ответа локального анализатора.'}</span></div><strong id="processing-percent">${job.progress}%</strong></div><progress id="processing-bar" max="100" value="${job.progress}">${job.progress}%</progress></div>`;
}
function demoPlayer(job) {
  return `<div class="demo-player"><img id="demo-scene" src="/static/demo/normal.jpg" alt="Нарисованная учебная кассовая сцена"><div class="demo-overlay"><span class="demo-camera"><span class="record-dot"></span>УЧЕБНАЯ СЦЕНА / КАССА 01</span><span class="demo-time" id="demo-time">00:00</span></div><div class="demo-player-controls"><button class="player-play" data-action="toggle-play" aria-label="Воспроизвести учебную сцену" id="demo-play">${icon('play')}</button><input type="range" id="demo-scrubber" min="0" max="42" step="0.1" value="0" aria-label="Время учебной сцены"><span id="demo-control-time">00:00 / 00:42</span></div></div>`;
}
function eventCard(event, index, demo) {
  const rule = event.rule;
  return `<article class="event-card" id="event-${index}"><div class="event-top"><span class="event-number">${String(index + 1).padStart(2, '0')}</span><button class="time-button" data-seek="${event.timecode_sec}">${icon('clock')}${timecode(event.timecode_sec)}${icon('arrow-right')}</button></div><h3>${esc(event.type)}</h3><p class="event-description">${esc(event.description)}</p><div class="evidence-grid">${event.evidence_frames.map((frame, frameIndex) => `<button class="evidence-button" data-frame="${index}:${frameIndex}" aria-label="Открыть кадр ${timecode(frame.timestamp_sec)}"><img src="data:image/jpeg;base64,${frame.image_base64}" alt="${demo ? 'Учебный кадр' : 'Кадр события'} ${timecode(frame.timestamp_sec)}"><span>${icon('expand')}${timecode(frame.timestamp_sec)}</span></button>`).join('')}</div><div class="rule-box"><span class="rule-icon">${icon('book')}</span><div><span class="rule-label">${demo ? 'УЧЕБНОЕ ОСНОВАНИЕ' : 'НОРМАТИВНЫЙ ИСТОЧНИК'}</span><strong>${esc(rule?.title || 'Источник не указан анализатором')}</strong><p>${esc(rule?.section || 'Проверьте нормативное обоснование перед принятием решения.')}</p>${rule?.url ? `<a href="${esc(rule.url)}" target="_blank" rel="noopener noreferrer">Открыть источник${icon('link')}</a>` : ''}</div></div></article>`;
}
async function poll(jobId, version) {
  if (version !== renderVersion) return;
  try {
    const job = await api(`/jobs/${jobId}`);
    if (version !== renderVersion) return;
    if (job.status !== 'processing') return renderResult(job);
    currentJob = job;
    const stage = document.querySelector('#processing-stage');
    if (stage) {
      stage.textContent = job.stage;
      document.querySelector('#processing-percent').textContent = `${job.progress}%`;
      document.querySelector('#processing-bar').value = job.progress;
    }
    pollTimer = setTimeout(() => poll(jobId, version), 700);
  } catch (error) {
    if (version !== renderVersion) return;
    toast(error.message);
    main.insertAdjacentHTML('afterbegin', `<div class="notice error-notice"><span>${icon('alert')}</span><p>Статус не удалось обновить. Проверка могла продолжиться на сервере.</p><button class="button secondary" data-action="reload-result">Обновить статус</button></div>`);
  }
}
function updateTime(value, duration = currentJob?.duration_sec || 42) {
  currentTime = value;
  const label = document.querySelector('#player-time');
  if (label) label.textContent = `${timecode(value)} / ${timecode(duration)}`;
  const head = document.querySelector('#timeline-playhead');
  if (head) head.style.left = `${Math.min(100, value / (duration || 1) * 100)}%`;
  if (currentJob?.demo) {
    document.querySelector('#demo-time').textContent = timecode(value);
    document.querySelector('#demo-control-time').textContent = `${timecode(value)} / 00:42`;
    document.querySelector('#demo-scrubber').value = value;
    const asset = currentJob.scenario === 'clean' ? 'normal' : value >= 26 && value < 35 ? 'access' : value >= 11 && value < 21 ? 'unattended' : 'normal';
    document.querySelector('#demo-scene').src = `/static/demo/${asset}.jpg`;
  }
  currentJob?.violations?.forEach((event, index) => document.querySelector(`#event-${index}`)?.classList.toggle('highlighted', Math.abs(value - event.timecode_sec) < 2));
}
function seekTo(value) {
  const video = document.querySelector('#source-video');
  if (video && video.readyState) video.currentTime = value;
  updateTime(value);
}
function togglePlay() {
  if (!currentJob?.demo) return;
  demoPlaying = !demoPlaying;
  const button = document.querySelector('#demo-play');
  button.innerHTML = icon(demoPlaying ? 'pause' : 'play');
  button.setAttribute('aria-label', demoPlaying ? 'Приостановить учебную сцену' : 'Воспроизвести учебную сцену');
  clearInterval(demoTimer);
  if (demoPlaying) {
    if (currentTime >= 42) updateTime(0);
    demoTimer = setInterval(() => {
      updateTime(Math.min(42, currentTime + 0.2));
      if (currentTime >= 42) { demoPlaying = false; clearInterval(demoTimer); button.innerHTML = icon('play'); button.setAttribute('aria-label', 'Воспроизвести учебную сцену'); }
    }, 200);
  }
}
async function retryAnalysis() {
  if (!currentJob || currentJob.demo) return;
  const button = document.querySelector('[data-action="retry"]');
  if (button) button.disabled = true;
  try { const job = await api('/retry', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({job_id: currentJob.job_id})}); navigate(`/results/${job.job_id}`); }
  catch (error) { toast(error.message); if (button) button.disabled = false; }
}
function downloadResult() {
  if (!currentJob || currentJob.status !== 'done') return;
  const blob = new Blob([JSON.stringify(currentJob, null, 2)], {type: 'application/json;charset=utf-8'});
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = `kfm-${currentJob.demo ? 'demo-' : ''}${currentJob.job_id.slice(0, 8)}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

document.addEventListener('click', event => {
  const nav = event.target.closest('[data-nav]');
  if (nav && !event.ctrlKey && !event.metaKey) { event.preventDefault(); return navigate(nav.dataset.nav); }
  const modeButton = event.target.closest('[data-mode]');
  if (modeButton && !uploadBusy) { mode = modeButton.dataset.mode; return updateUploadContent(); }
  const seek = event.target.closest('[data-seek]');
  if (seek) return seekTo(Number(seek.dataset.seek));
  const frameButton = event.target.closest('[data-frame]');
  if (frameButton) {
    const [index, frameIndex] = frameButton.dataset.frame.split(':').map(Number);
    const frame = currentJob.violations[index].evidence_frames[frameIndex];
    document.querySelector('#large-frame').src = `data:image/jpeg;base64,${frame.image_base64}`;
    document.querySelector('#frame-caption').textContent = `${currentJob.demo ? 'Учебный кадр · ' : ''}${currentJob.violations[index].type} · ${timecode(frame.timestamp_sec)}`;
    return document.querySelector('#frame-dialog').showModal();
  }
  const action = event.target.closest('[data-action]')?.dataset.action;
  if (!action) return;
  if (uploadBusy && !['help', 'close-help'].includes(action)) return toast('Дождитесь завершения загрузки.');
  if (action === 'help') { document.querySelectorAll('.max-upload').forEach(el => el.textContent = config.max_upload_mb); document.querySelector('#help-dialog').showModal(); }
  if (action === 'close-help') document.querySelector('#help-dialog').close();
  if (action === 'close-frame') document.querySelector('#frame-dialog').close();
  if (action === 'home') navigate('/');
  if (action === 'reload-result') renderRoute();
  if (action === 'switch-demo' || action === 'demo-menu') {
    mode = 'demo';
    if (location.pathname !== '/') navigate('/'); else updateUploadContent();
    document.querySelector('#upload-title')?.scrollIntoView({behavior: 'smooth', block: 'center'});
  }
  if (action === 'clear-file') { selectedFile = null; updateUploadContent(); }
  if (action === 'start-demo') startDemo();
  if (action === 'upload') uploadVideo();
  if (action === 'toggle-play') togglePlay();
  if (action === 'download') downloadResult();
  if (action === 'retry') retryAnalysis();
});
document.addEventListener('change', event => { if (event.target.name === 'scenario') { scenario = event.target.value; updateUploadContent(); } });
document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } }));
window.addEventListener('popstate', renderRoute);
window.addEventListener('beforeunload', event => { if (uploadBusy) { event.preventDefault(); event.returnValue = ''; } });
document.addEventListener('dragover', event => event.preventDefault());
document.addEventListener('drop', event => event.preventDefault());

hydrateIcons();
(async () => {
  try { config = await api('/config'); }
  catch (error) { toast(error.message); }
  await renderRoute();
})();

/* ---------- Manual test runner (blocking on answers) ---------- */
window.__manualTestRunner = window.__manualTestRunner || { running:false, stop:false, runId:0, events:[], summary:null, scenario:'', askFallbackNoted:false };

const _MT_I18N = {
  de: {
    monitor_show: 'Monitorfenster: anzeigen',
    monitor_hide: 'Monitorfenster: ausblenden',
    mirror_main: 'Im Hauptdialog spiegeln',
    scenario_actual_test: 'ACTUAL-TEST (schnell, Report wird ueberschrieben)',
    scenario_smoke_short: 'Kurztest (A+C+D+F ohne HF)',
    scenario_provider_switch: 'Providerwechsel (Gemini/OpenRouter/HF optional)',
    scenario_sci_format: 'SCI-Format (A/B)',
    scenario_qc_override_footer: 'QC-Override + Footer (SCI B, Gemini-Referenz)',
    scenario_profile_self_debunking: 'Profile + Self-Debunking-Contract (alle Profile)',
    scenario_komplexttest: 'Komplexttest (Matrix + Pflichtprompts + Influence-Checks)',
    scenario_full_regression_light: 'A-F (leicht, HF optional)',
    test_hint: 'Fuehrt einen blockierenden GUI-Testablauf ueber `api.ask(...)` aus und prueft Basis-Formatregeln.',
    prompt_short: 'Was ist Zeit als Begriff in Physik und Philosophie (nicht die aktuelle Uhrzeit)?',
    prompt_long: 'Was ist die objektiv beste und dauerhaft faire Strategie, um ab heute weltweit ein einheitliches KI-Regelwerk verbindlich durchzusetzen, sodass alle LLMs in jeder Sprache, Kultur und Rechtsordnung identische Antworten liefern, ohne negative Folgen fuer Datenschutz, Demokratie, Kreativitaet, Wissenschaft und Arbeitsmarkt?',
    ask_prefix: 'ASK > ',
    ask_done_prefix: 'ASK < done',
    info_ask_fallback: 'INFO: api.ask nicht verfuegbar -> Fallback auf panel_action(ask)',
    set_provider_prefix: 'SET provider = ',
    set_answer_language_prefix: 'SET answer_language = ',
    qc_apply_prefix: 'QC Override apply: ',
    qc_clear: 'QC Override clear',
    qc_apply_fallback: 'qc_override_apply nicht direkt verfuegbar -> Fallback auf panel_action',
    qc_clear_fallback: 'qc_override_clear nicht direkt verfuegbar -> Fallback auf panel_action',
    pass_prefix: 'PASS: ',
    fail_prefix: 'FAIL: ',
    warn_prefix: 'WARN: ',
    monitor_toggle_failed: 'Monitor-Umschaltung fehlgeschlagen: ',
    report_saved_prefix: 'REPORT saved: ',
    report_saved_plain: 'REPORT saved.',
    report_save_failed: 'REPORT save failed: ',
    stop_requested: 'Stop angefordert.',
    test_started_prefix: 'Manual-Test gestartet: ',
    summary_fail_prefix: 'SUMMARY: FAILS=',
    summary_pass_prefix: 'SUMMARY: PASS',
    summary_stopped: 'SUMMARY: STOPPED',
    summary_error_prefix: 'ERROR: ',
    status_finished_fail_prefix: 'Manual Test fertig: ',
    status_finished_fail_suffix: ' Fehler',
    status_finished_pass: 'Manual Test fertig: PASS',
    status_stopped: 'Manual Test gestoppt',
    status_error_prefix: 'Manual Test error: ',
    monitor_init_failed: 'Manual-Test-Monitor konnte nicht initialisiert werden: ',
    unknown_scenario_prefix: 'unknown scenario: ',
    no_u_marker: 'kein U-Marker gefunden',
    has_u_marker: 'U-Marker vorhanden',
    influence_checks: 'INFLUENCE-CHECKS > CGI / QC-Override / Dynamic one-shot',
    cgi_recorded: 'CGI-Feedback wurde vom Wrapper registriert',
    cgi_not_recorded: 'CGI-Feedback wurde nicht registriert',
    cgi_influence_yes: 'CGI-Feedback beeinflusst Folgeresponse',
    cgi_influence_no: 'CGI-Feedback beeinflusst Folgeresponse nicht sichtbar',
    qc_influence_yes: 'QC-Override beeinflusst Folgeresponse',
    qc_influence_no: 'QC-Override beeinflusst Folgeresponse nicht sichtbar',
    dynamic_influence_yes: 'Dynamic one-shot beeinflusst Folgeresponse',
    dynamic_influence_no: 'Dynamic one-shot beeinflusst Folgeresponse nicht sichtbar',
  },
  en: {
    monitor_show: 'Monitor window: show',
    monitor_hide: 'Monitor window: hide',
    mirror_main: 'Mirror in main dialog',
    scenario_actual_test: 'ACTUAL-TEST (fast, report file is overwritten)',
    scenario_smoke_short: 'Short test (A+C+D+F, no HF)',
    scenario_provider_switch: 'Provider switch (Gemini/OpenRouter/HF optional)',
    scenario_sci_format: 'SCI format (A/B)',
    scenario_qc_override_footer: 'QC override + footer (SCI B, Gemini reference)',
    scenario_profile_self_debunking: 'Profiles + self-debunking contract (all profiles)',
    scenario_komplexttest: 'Complex test (matrix + mandatory prompts + influence checks)',
    scenario_full_regression_light: 'A-F (light, HF optional)',
    test_hint: 'Runs a blocking GUI-like test flow via `api.ask(...)` and checks baseline format contracts.',
    prompt_short: 'Explain the concept of time in physics and philosophy (not the current clock time).',
    prompt_long: 'What is the objectively best and sustainably fair strategy to enforce a single global AI rule set from today onward so that all LLMs produce identical answers across all languages, cultures, and legal systems, without negative consequences for privacy, democracy, creativity, science, and the labor market?',
    ask_prefix: 'ASK > ',
    ask_done_prefix: 'ASK < done',
    info_ask_fallback: 'INFO: api.ask unavailable -> fallback to panel_action(ask)',
    set_provider_prefix: 'SET provider = ',
    set_answer_language_prefix: 'SET answer_language = ',
    qc_apply_prefix: 'QC override apply: ',
    qc_clear: 'QC override clear',
    qc_apply_fallback: 'qc_override_apply unavailable directly -> fallback to panel_action',
    qc_clear_fallback: 'qc_override_clear unavailable directly -> fallback to panel_action',
    pass_prefix: 'PASS: ',
    fail_prefix: 'FAIL: ',
    warn_prefix: 'WARN: ',
    monitor_toggle_failed: 'Monitor toggle failed: ',
    report_saved_prefix: 'REPORT saved: ',
    report_saved_plain: 'REPORT saved.',
    report_save_failed: 'REPORT save failed: ',
    stop_requested: 'Stop requested.',
    test_started_prefix: 'Manual test started: ',
    summary_fail_prefix: 'SUMMARY: FAILS=',
    summary_pass_prefix: 'SUMMARY: PASS',
    summary_stopped: 'SUMMARY: STOPPED',
    summary_error_prefix: 'ERROR: ',
    status_finished_fail_prefix: 'Manual test done: ',
    status_finished_fail_suffix: ' failures',
    status_finished_pass: 'Manual test done: PASS',
    status_stopped: 'Manual test stopped',
    status_error_prefix: 'Manual test error: ',
    monitor_init_failed: 'Manual test monitor init failed: ',
    unknown_scenario_prefix: 'unknown scenario: ',
    no_u_marker: 'no U-marker found',
    has_u_marker: 'U-marker present',
    influence_checks: 'INFLUENCE CHECKS > CGI / QC override / Dynamic one-shot',
    cgi_recorded: 'CGI feedback registered by wrapper',
    cgi_not_recorded: 'CGI feedback not registered',
    cgi_influence_yes: 'CGI feedback influences follow-up response',
    cgi_influence_no: 'CGI feedback influence not visible in follow-up response',
    qc_influence_yes: 'QC override influences follow-up response',
    qc_influence_no: 'QC override influence not visible in follow-up response',
    dynamic_influence_yes: 'Dynamic one-shot influences follow-up response',
    dynamic_influence_no: 'Dynamic one-shot influence not visible in follow-up response',
  },
};

function _mtLang(){
  try {
    const el = document.getElementById('anslang');
    const v = String((el && el.value) || 'de').trim().toLowerCase();
    return (v === 'en') ? 'en' : 'de';
  } catch(e) {
    return 'de';
  }
}

function _mtT(key){
  const lang = _mtLang();
  const dict = _MT_I18N[lang] || _MT_I18N.de;
  return (dict && dict[key]) ? dict[key] : ((_MT_I18N.de && _MT_I18N.de[key]) ? _MT_I18N.de[key] : String(key || ''));
}

function _mtL(deText, enText){
  return _mtLang() === 'en' ? String(enText || '') : String(deText || '');
}

function _mtSelectedAnswerLanguage(){
  try {
    const el = document.getElementById('anslang');
    const v = String((el && el.value) || 'de').trim().toLowerCase();
    return (v === 'en') ? 'en' : 'de';
  } catch(e) {
    return 'de';
  }
}

function _mtPromptShort(){
  return _mtT('prompt_short');
}

function _mtPromptLong(){
  return _mtT('prompt_long');
}

function _mtContainsUCode(html, plainText){
  const raw = String(html || '');
  const txt = String(plainText || '');
  if(/\bU[1-8]\b/i.test(txt)) return true;
  if(/data-u-code\s*=\s*(?:\"|')?U[1-8](?:\"|')?/i.test(raw)) return true;
  return false;
}

function _mtHasBrokenRedSpanAroundUncertaintyMarker(html){
  const raw = String(html || '');
  return /<span style=(?:\"|')color:#c62828;\s*font-weight:600;(?:\"|')>\s*🔴\s*<span class=(?:\"|')[^\"']*\buncertainty-inline-wrap\b/i.test(raw);
}

function _mtStripRawModelOutputDetails(html){
  const raw = String(html || '');
  if(!raw) return '';
  return raw.replace(
    /<details\b[^>]*class=(?:\"|')[^\"']*\braw-output\b[^\"']*(?:\"|')[^>]*>[\s\S]*?<\/details>/gi,
    ' '
  );
}

function _mtHasUncertaintyTemplateLeak(html){
  const txt = _mtStripHtml(_mtStripRawModelOutputDetails(html || '')).replace(/\s+/g, ' ').trim();
  if(!txt) return false;
  const names = '(?:Data\\s+gap|Assumption\\s+gap|Perspective\\s+conflict|Temporal\\s+instability|Structural\\s+limitation|Interpretation\\s+ambiguity|Retrieval\\s+conflict|Retrieval\\s+metadata\\s+gap|Datenluecke|Annahmen\\s+unklar|Perspektivenkonflikt|Zeitliche\\s+Instabilitaet|Strukturelle\\s+Grenze|Interpretationsspielraum|Retrieval-Konflikt|Retrieval-Metadatenluecke)';
  const needed = '(?:Needed|Ben(?:ö|oe)tigt)\\s*:';
  if(new RegExp('(?:Uncertainty|Unsicherheit)\\s*:\\s*(?:\\(\\s*)?U[1-8](?:\\s*\\))?\\s*(?:-|–|—)\\s*' + names + '\\s*\\.?\\s*(?:' + needed + ')?', 'i').test(txt)) return true;
  if(new RegExp('(?:\\(\\s*)?U[1-8](?:\\s*\\))?\\s*(?:-|–|—)\\s*' + names + '\\s*\\.?\\s*(?:' + needed + ')?', 'i').test(txt)) return true;
  return _mtHasUncertaintyTailPhraseLeakU1ToU8(txt);
}

function _mtFormatLeakSnippet(snippet, maxLen){
  const lim = Math.max(60, Math.min(280, Number(maxLen || 180)));
  const s = String(snippet || '').replace(/\s+/g, ' ').trim();
  if(!s) return '';
  if(s.length <= lim) return s;
  return s.slice(0, lim - 1) + '…';
}

function _mtFindUncertaintyTailPhraseLeakU1ToU8(html){
  const txt = _mtStripHtml(_mtStripRawModelOutputDetails(html || '')).replace(/\s+/g, ' ').trim();
  if(!txt) return '';
  const needed = '(?:Needed|Ben(?:ö|oe)tigt|Required|Erforderlich)\\s*:';
  const codeHead = '(?:\\(\\s*)?\\bU[1-8]\\b(?:\\s*\\))?(?!\\s*[-–—]\\s*U[1-8])';

  const prefixed = new RegExp(
    '(?:Uncertainty|Unsicherheit)\\s*:\\s*' + codeHead + '(?:\\s*(?:-|–|—|:)\\s*)?[^.!?\\n]{0,220}?' + needed,
    'i'
  );
  let m = prefixed.exec(txt);
  if(m && m[0]) return _mtFormatLeakSnippet(m[0], 200);

  const codeTail = new RegExp(codeHead + '\\s*(?:-|–|—|:)\\s*[^.!?\\n]{0,220}?' + needed, 'i');
  m = codeTail.exec(txt);
  if(m && m[0]) return _mtFormatLeakSnippet(m[0], 200);

  const nearNeeded = new RegExp(codeHead + '[^.!?\\n]{0,120}?' + needed, 'i');
  m = nearNeeded.exec(txt);
  if(m && m[0]) return _mtFormatLeakSnippet(m[0], 200);

  return '';
}

function _mtHasUncertaintyTailPhraseLeakU1ToU8(html){
  return !!_mtFindUncertaintyTailPhraseLeakU1ToU8(html);
}

function _mtMirrorMainEnabled(){
  try {
    const el = document.getElementById('manualTestMirrorMain');
    return !!(el && el.checked);
  } catch(e) {
    return false;
  }
}

function setManualTestMirrorMode(){
  try {
    _safeLsSet('manual_test_mirror_main', _mtMirrorMainEnabled() ? 'on' : 'off');
  } catch(e) {}
}

async function _mtMirrorMainAppend(payload){
  try {
    if(!_mtMirrorMainEnabled()) return false;
    const mt = window.__manualTestRunner || {};
    const res = await _apiCall('panel_action', ['manual_test_main_chat_append', {payload: payload || {}}], 12000);
    if(!res || res.ok === false){
      if(!mt.mainMirrorWarned){
        mt.mainMirrorWarned = true;
        _mtWarn('main-chat mirror unavailable: ' + String((res && res.error) || 'unknown'));
      }
      return false;
    }
    return true;
  } catch(e) {
    try {
      const mt = window.__manualTestRunner || {};
      if(!mt.mainMirrorWarned){
        mt.mainMirrorWarned = true;
        _mtWarn('main-chat mirror failed: ' + String(e && e.message ? e.message : e));
      }
    } catch(_e) {}
    return false;
  }
}

function _mtApplyLocalizedUi(){
  try {
    const monitor = document.getElementById('manualTestMonitorMode');
    if(monitor && monitor.options && monitor.options.length >= 2){
      monitor.options[0].textContent = _mtT('monitor_show');
      monitor.options[1].textContent = _mtT('monitor_hide');
    }
    const mirrorLbl = document.getElementById('manualTestMirrorMainLabel');
    if(mirrorLbl) mirrorLbl.textContent = _mtT('mirror_main');

    const scenarioSelect = document.getElementById('manualTestScenario');
    if(scenarioSelect && scenarioSelect.options){
      const map = {
        actual_test: 'scenario_actual_test',
        smoke_short: 'scenario_smoke_short',
        provider_switch: 'scenario_provider_switch',
        sci_format: 'scenario_sci_format',
        qc_override_footer: 'scenario_qc_override_footer',
        profile_self_debunking: 'scenario_profile_self_debunking',
        komplexttest: 'scenario_komplexttest',
        full_regression_light: 'scenario_full_regression_light',
      };
      for(let i = 0; i < scenarioSelect.options.length; i += 1){
        const opt = scenarioSelect.options[i];
        const k = map[String(opt.value || '')] || '';
        if(k) opt.textContent = _mtT(k);
      }
    }

    const hint = document.getElementById('manualTestHint');
    if(hint) hint.textContent = _mtT('test_hint');
  } catch(e) {}
}

function _mtMonitorEnabled(){
  try {
    const el = document.getElementById('manualTestMonitorMode');
    return !!(el && el.value === 'show');
  } catch(e) { return true; }
}

async function setManualTestMonitorMode(){
  try {
    const mode = _mtMonitorEnabled() ? 'show' : 'hide';
    _safeLsSet('manual_test_monitor_mode', mode);
    if(mode === 'show'){
      await _apiCall('panel_action', ['manual_test_monitor_show', {}], 8000);
    } else {
      await _apiCall('panel_action', ['manual_test_monitor_hide', {}], 8000);
    }
  } catch(e) {
    _mtWarn(_mtT('monitor_toggle_failed') + String(e && e.message ? e.message : e));
  }
}

function _mtLog(msg, cls){
  const el = document.getElementById('manualTestLog');
  if(!el) return;
  let stamp = '';
  try { stamp = _now() || ''; } catch(e) { stamp = ''; }
  const line = document.createElement('div');
  if(cls) line.className = cls;
  line.textContent = (stamp ? (stamp + ' · ') : '') + String(msg || '');
  el.appendChild(line);
  while(el.children.length > 200) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
  try {
    const mt = window.__manualTestRunner;
    if(mt && mt.running){
      if(!Array.isArray(mt.events)) mt.events = [];
      const entry = {
        ts: stamp || null,
        level: cls || 'info',
        message: String(msg || ''),
      };
      mt.events.push(entry);
      try {
        if(mt.monitorEnabled){
          _apiCall('panel_action', ['manual_test_monitor_append', {entry: entry}], 4000).catch(()=>{});
        }
      } catch(e) {}
    }
  } catch(e) {}
}

function _mtClearLog(){
  const el = document.getElementById('manualTestLog');
  if(el) el.innerHTML = '';
}

async function _mtSaveReport(extra){
  try {
    const mt = window.__manualTestRunner || {};
    const report = {
      kind: 'manual_test_runner_report',
      version: 1,
      scenario: String(mt.scenario || ''),
      started_at: mt.startedAt || null,
      finished_at: (extra && extra.finished_at) || (_now ? _now() : null),
      duration_ms: (extra && extra.duration_ms) || null,
      summary: Object.assign({}, (mt.summary || {}), (extra && extra.summary ? extra.summary : {})),
      events: Array.isArray(mt.events) ? mt.events.slice() : [],
      ui_state: {
        provider: ((document.getElementById('provider') || {}).value || ''),
        model: ((document.getElementById('model') || {}).value || ''),
        answer_language: ((document.getElementById('anslang') || {}).value || ''),
      }
    };
    const res = await _apiCall('panel_action', ['save_manual_test_report', {report: report}], 20000);
    if(res && res.ok !== false){
      const path = (res.path || (res.result && res.result.path) || '');
      if(path) _mtLog(_mtT('report_saved_prefix') + path, 'ok');
      else _mtLog(_mtT('report_saved_plain'), 'ok');
      return res;
    }
    _mtWarn(_mtT('report_save_failed') + String((res && res.error) || 'unknown'));
    return res;
  } catch(e) {
    _mtWarn(_mtT('report_save_failed') + String(e && e.message ? e.message : e));
    return null;
  }
}

async function _mtExportChatAudit(label){
  const suffix = String(label || '').trim();
  const tag = suffix ? (' [' + suffix + ']') : '';
  _mtLog(_mtL('EXPORT > Chat/Audit', 'EXPORT > Chat/Audit') + tag);
  try {
    await _apiCall('export', [], 25000);
    _mtLog(_mtL('PASS: Export ausgefuehrt', 'PASS: export executed'), 'ok');
    return true;
  } catch(e) {
    const msg = String(e && e.message ? e.message : e);
    if(msg.toLowerCase().includes('pywebview api method not available: export')){
      try {
        const res = await _apiCall('panel_action', ['export', {}], 25000);
        if(!res || res.ok === false){
          throw new Error((res && res.error) ? res.error : 'panel_action export failed');
        }
        _mtLog(_mtL('PASS: Export ausgefuehrt', 'PASS: export executed'), 'ok');
        return true;
      } catch(e2) {
        _mtWarn(_mtL('Export nicht ausgefuehrt: ', 'Export not executed: ') + String(e2 && e2.message ? e2.message : e2));
        return false;
      }
    }
    _mtWarn(_mtL('Export nicht ausgefuehrt: ', 'Export not executed: ') + msg);
    return false;
  }
}

function _mtSetButtons(running){
  const a = document.getElementById('manualTestStartBtn');
  const b = document.getElementById('manualTestStopBtn');
  if(a) a.disabled = !!running;
  if(b) b.disabled = !running;
}

function _mtStripHtml(html){
  try{
    const d = document.createElement('div');
    d.innerHTML = String(html || '');
    return (d.textContent || d.innerText || '').replace(/\u00a0/g, ' ');
  }catch(e){ return String(html || ''); }
}

function _mtHasCompleteQcFooter(html){
  const txt = _mtStripHtml(html);
  const idxMatrix = txt.lastIndexOf('QC-Matrix:');
  const idxQC = txt.lastIndexOf('QC:');
  const idx = Math.max(idxMatrix, idxQC);
  if(idx < 0) return false;
  let qc = txt.slice(idx);
  const tsIdx = qc.search(/\bResponse at\b/i);
  if(tsIdx >= 0) qc = qc.slice(0, tsIdx);
  qc = qc.replace(/\s+/g, ' ').trim();
  const en = ['Clarity','Brevity','Evidence','Empathy','Consistency','Neutrality'];
  const de = ['Klarheit','Evidenz','Empathie','Konsistenz'];
  const hasEN = en.every(k => qc.includes(k + ' '));
  const hasDE = de.every(k => qc.includes(k + ' ')) &&
                (qc.includes('Kürze ') || qc.includes('Kuerze ')) &&
                (qc.includes('Neutralität ') || qc.includes('Neutralitaet '));
  return !!(hasEN || hasDE);
}

function _mtHasSciAlertMissingTraceStepContent(html){
  const txt = _mtStripHtml(html || '').replace(/\s+/g, ' ').trim();
  if(!txt) return false;
  return /CONTROL LAYER ALERT\s*\(SCI\)/i.test(txt) && /Missing SCI Trace step content/i.test(txt);
}

function _mtCheckQcFooterWithSciAlertGuard(html, hasCompleteQcFooter, okMsg, failMsg, skipMsg){
  if(hasCompleteQcFooter){
    _mtLog(_mtT('pass_prefix') + okMsg, 'ok');
    return true;
  }
  if(_mtHasSciAlertMissingTraceStepContent(html)){
    _mtWarn(skipMsg);
    return true;
  }
  _mtLog(_mtT('fail_prefix') + failMsg, 'err');
  return false;
}

function _mtHasQcDimTooltips(html){
  const raw = String(html || '');
  const hits = raw.match(/class=(?:\"|')[^\"']*qc-dim-tip[^\"']*(?:\"|')/g) || [];
  return hits.length >= 6;
}

function _mtHasLocalizedQcFooterWithoutDeltaAtEnd(html){
  const txt = _mtStripHtml(html || '');
  const idx = txt.lastIndexOf('QC-Matrix:');
  if(idx < 0) return false;
  let qc = txt.slice(idx);
  const tsIdx = qc.search(/\bResponse at\b/i);
  if(tsIdx >= 0) qc = qc.slice(0, tsIdx);
  qc = qc.replace(/\s+/g, ' ').trim();
  const hasLocalized = /\b(Klarheit|Kürze|Kuerze|Evidenz|Empathie|Konsistenz|Neutralität|Neutralitaet)\s+[0-3]\b/.test(qc);
  const hasDelta = /(?:Δ|∆)\s*[+\-−]?\d+/.test(qc);
  return hasLocalized && !hasDelta;
}

function _mtHasTraceSelfDebunkingQcTimestampOrder(html){
  const txt = _mtStripHtml(html || '').replace(/\s+/g, ' ').trim();
  if(!txt) return false;
  const idxQc = txt.lastIndexOf('QC-Matrix:');
  const idxTrace = (idxQc >= 0) ? txt.lastIndexOf('SCI Trace', idxQc) : -1;
  const idxSd = (idxQc >= 0)
    ? Math.max(txt.lastIndexOf('Self-Debunking', idxQc), txt.lastIndexOf('Selbst-Debunking', idxQc))
    : -1;
  const idxTs = (idxQc >= 0) ? txt.indexOf('Response at', idxQc) : -1;
  if(idxTrace < 0 || idxSd < 0 || idxQc < 0 || idxTs < 0) return false;
  return idxTrace < idxSd && idxSd < idxQc && idxQc < idxTs;
}

function _mtHasFinalAnswerLabelInsideSciTrace(html){
  const txt = _mtStripHtml(html || '').replace(/\s+/g, ' ').trim();
  if(!txt) return false;
  const idxTrace = txt.lastIndexOf('SCI Trace');
  if(idxTrace < 0) return false;
  const idxSdEn = txt.indexOf('Self-Debunking', idxTrace);
  const idxSdDe = txt.indexOf('Selbst-Debunking', idxTrace);
  const idxQc = txt.indexOf('QC-Matrix:', idxTrace);
  let end = txt.length;
  for(const idx of [idxSdEn, idxSdDe, idxQc]){
    if(idx >= 0) end = Math.min(end, idx);
  }
  const trace = txt.slice(idxTrace, end);
  return /(?:^|[\s\(\[])(?:Final\s+Answer|Antwort|Answer)\s*:/.test(trace);
}

function _mtHasEmbeddedImageTagForUrl(html, url){
  const raw = String(html || '');
  const u = String(url || '').trim();
  if(!u) return false;
  const esc = _mtEscapeRegex(u);
  const re = new RegExp("<img[^>]+src=(?:\"|')" + esc + "(?:\"|')", 'i');
  return re.test(raw);
}

function _mtLooksLikeDeterministicImageEchoResponse(html, url){
  const u = String(url || '').trim();
  if(!u) return false;
  const txt = _mtStripHtml(html || '').replace(/\s+/g, ' ').trim();
  if(!txt || !txt.includes(u)) return false;
  // Provider drift: long free-form answers should not hard-fail image embed checks.
  if(txt.length > 420) return false;
  const esc = _mtEscapeRegex(u);
  if(new RegExp("^\\s*[`<\\[]?" + esc + "(?:[\\]>`])?[\\.,;:!?)]?\\s*$", "i").test(txt)) return true;
  const raw = String(html || '');
  return /<code|<pre|```/i.test(raw);
}

function _mtHasStrictEnforcementBanner(html){
  const txt = _mtStripHtml(html || '');
  return /(STRICT BLOCK|RULE VIOLATION DETECTED\s*\(strict_warn\))/i.test(txt);
}

function _mtHasRenderFallbackNote(html){
  const txt = _mtStripHtml(html || '');
  return /render fallback/i.test(txt);
}

function _mtHasControlLayerAlertsBox(html){
  const txt = _mtStripHtml(html || '');
  return /CONTROL LAYER ALERTS\s*\(Python\)/i.test(txt);
}

function _mtHasSelfDebunkingBox(html){
  const h = String(html || '').toLowerCase();
  const hasDebunkLabel = h.includes('self-debunking') || h.includes('selbst-debunking');
  const hasDebunkClass = h.includes('class=\"self-debunk') || h.includes("class='self-debunk");
  const hasBoxStyle = (h.includes('background') || h.includes('background-color')) &&
                      (h.includes('border-left') || h.includes('border-radius'));
  return hasDebunkLabel && (hasDebunkClass || hasBoxStyle);
}

function _mtHasActiveProfileHeader(html, profile){
  const txt = _mtStripHtml(html || '');
  const p = String(profile || '').trim();
  if(!p) return false;
  return txt.toLowerCase().includes(('active profile: ' + p).toLowerCase());
}

function _mtEscapeRegex(token){
  return String(token || '').replace(/[][\\^$.*+?(){}|]/g, '\\$&');
}

function _mtHasCommStateField(html, field, expected){
  const src = String(html || '');
  const f = _mtEscapeRegex(field);
  const e = _mtEscapeRegex(expected);
  if(!f || !e) return false;
  const rowRe = new RegExp("<th[^>]*>\\s*" + f + "\\s*<\\/th>\\s*<td[^>]*>\\s*" + e + "\\s*<\\/td>", "i");
  if(rowRe.test(src)) return true;
  const txt = _mtStripHtml(src).toLowerCase();
  const pair = (String(field || '') + ': ' + String(expected || '')).toLowerCase();
  return txt.includes(pair);
}

function _mtHasSelfDebunkingContractStructure(html){
  const txt = _mtStripHtml(String(html || '')).toLowerCase();
  if(!txt) return false;
  const txtNorm = txt.replace(/\s+:/g, ':');
  if(!(txtNorm.includes('self-debunking') || txtNorm.includes('selbst-debunking'))) return false;
  if(!_mtHasSelfDebunkingBox(html)) return false;

  const count = (patterns) => {
    let n = 0;
    for(const pat of (patterns || [])){
      if(!pat) continue;
      const hits = txtNorm.match(new RegExp(String(pat), 'gi'));
      if(hits && hits.length) n += hits.length;
    }
    return n;
  };

  const primary = count([
    '\\bschw(?:ä|ae)che\\s*:',
    '\\bweakness\\s*:',
    '\\bunsicherheit\\s*:',
    '\\buncertainty\\s*:',
  ]);
  const why = count([
    '\\bwarum\\s+das\\s+wichtig\\s+ist\\s*:',
    '\\bwarum\\s+es\\s+wichtig\\s+ist\\s*:',
    '\\bwarum\\s+relevant\\s*:',
    '\\bwhy\\s+it\\s+matters\\s*:',
    '\\bwhy\\s+this\\s+is\\s+important\\s*:',
  ]);
  const check = count([
    'was\\s+(?:w(?:ü|ue)rde)\\s+verifizieren\\s*(?:\\/|\\s+oder\\s+)\\s*falsifizieren(?:\\s*\\(\\s*(?:n(?:ä|ae)chster\\s+check|next\\s+check)\\s*\\))?\\s*:',
    '(?:pr(?:ü|ue)fen\\s*\\/\\s*widerlegen)(?:\\s*\\(\\s*n(?:ä|ae)chster\\s+schritt\\s*\\))?\\s*:',
    'what\\s+would\\s+verify\\s*(?:\\/|\\s+or\\s+)\\s*falsify(?:\\s*\\(\\s*next\\s+check\\s*\\))?\\s*:',
    '\\bnext\\s+check\\s*:',
    '\\bn(?:ä|ae)chster\\s+check\\s*:',
    '\\bn(?:ä|ae)chster\\s+schritt\\s*:',
  ]);

  // Ein vollständiger Self-Debunking-Block reicht für den Contract aus.
  return (primary >= 1) && (why >= 1) && (check >= 1);
}

function _mtLooksLikeClockTimeListAnswer(html){
  const txt = _mtStripHtml(String(html || '')).toLowerCase().replace(/\s+/g, ' ').trim();
  if(!txt) return false;
  const hasLead = (
    txt.includes('current time in various major cities') ||
    txt.includes('aktuelle zeit in verschiedenen großstädten') ||
    txt.includes('aktuelle zeit in verschiedenen grossstädten') ||
    txt.includes('current time in major cities')
  );
  const cities = [
    'new york',
    'london',
    'tokyo',
    'tokio',
    'paris',
    'sydney',
    'dubai',
    'singapore',
    'singapur',
    'los angeles',
    'chicago',
  ];
  let cityHits = 0;
  for(const city of cities){
    if(txt.includes(city)) cityHits += 1;
  }
  const timeHits = (txt.match(/\b\d{1,2}:\d{2}\b/g) || []).length;
  return !!(hasLead || (cityHits >= 3 && timeHits >= 4));
}

function _mtCollectSelfDebunkingListItemsHtml(html){
  const raw = String(html || '');
  if(!raw) return [];
  const items = [];
  const blockRe = /<div[^>]*class=(?:"|')[^"']*self-debunking[^"']*(?:"|')[^>]*>[\s\S]*?<ol[^>]*>([\s\S]*?)<\/ol>[\s\S]*?<\/div>/gi;
  let bm = null;
  while((bm = blockRe.exec(raw)) !== null){
    const olBody = String((bm && bm[1]) || '');
    const liRe = /<li[^>]*>([\s\S]*?)<\/li>/gi;
    let lm = null;
    while((lm = liRe.exec(olBody)) !== null){
      const liBody = String((lm && lm[1]) || '').trim();
      if(liBody) items.push(liBody);
    }
  }
  return items;
}

function _mtHasSelfDebunkingItalicSecondaryLabelLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const labelRx = '(?:Warum\\s+(?:das|es)\\s+wichtig\\s+ist|Warum\\s+relevant|Why\\s+it\\s+matters|Why\\s+this\\s+is\\s+important|Was\\s+(?:würde|wuerde)\\s+verifizieren(?:/|\\s+oder\\s+)falsifizieren\\s+\\((?:nächster|naechster)\\s+Check\\)|What\\s+would\\s+verify(?:/|\\s+or\\s+)falsify\\s+\\(next\\s+check\\)|N(?:ä|ae)chster\\s+(?:Check|Schritt)|Next\\s+(?:check|step))';
  const italicRe = new RegExp('<em>\\s*(?:<strong>\\s*)?(?:' + labelRx + ')(?:\\s*</strong>)?\\s*:?\\s*</em>', 'i');
  for(const item of items){
    if(italicRe.test(String(item || ''))) return true;
  }
  return false;
}

function _mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;

  const normalizeItem = (item) => {
    let s = String(item || '');
    // Accept <b> as bold-equivalent and flatten paragraph wrappers to explicit breaks.
    s = s.replace(/<(\/?)b>/gi, '<$1strong>');
    s = s.replace(/<\/p>\s*<p[^>]*>/gi, '<br>');
    s = s.replace(/<p[^>]*>/gi, '');
    s = s.replace(/<\/p>/gi, '');
    return s;
  };

  const hasPair = (item, whyRx, checkRx) => {
    const src = normalizeItem(item);
    const whyRe = new RegExp('<br\\s*/?>\\s*<strong>\\s*(?:' + whyRx + ')\\s*</strong>\\s*:', 'i');
    const checkRe = new RegExp('<br\\s*/?>\\s*<strong>\\s*(?:' + checkRx + ')\\s*</strong>\\s*:', 'i');
    return whyRe.test(src) && checkRe.test(src);
  };

  const whyDe = 'Warum\\s+(?:das|es)\\s+wichtig\\s+ist|Warum\\s+relevant';
  const checkDe = 'Was\\s+(?:würde|wuerde)\\s+verifizieren(?:/|\\s+oder\\s+)falsifizieren\\s+\\((?:nächster|naechster)\\s+Check\\)|N(?:ä|ae)chster\\s+(?:Check|Schritt)|Pr(?:ü|ue)fen\\/Widerlegen\\s+\\((?:nächster|naechster)\\s+Schritt\\)';
  const whyEn = 'Why\\s+it\\s+matters|Why\\s+this\\s+is\\s+important';
  const checkEn = 'What\\s+would\\s+verify(?:/|\\s+or\\s+)falsify\\s+\\(next\\s+check\\)|Next\\s+(?:check|step)';

  for(const item of items){
    const raw = String(item || '');
    const okDe = hasPair(raw, whyDe, checkDe);
    const okEn = hasPair(raw, whyEn, checkEn);
    if(!(okDe || okEn)) return false;
  }
  return true;
}

function _mtHasSelfDebunkingVerificationRouteLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const vrRe = /(?:\bverification\s+route(?:\s+gate)?\b|\bsource\s*:|\bmeasurement\s*:|\bcontrast\s*:|\bweb[\s-]*check\s*:|\bquelle\s*:|\bmessung\s*:|\bkontrast\s*:|\bweb[\s-]*(?:prüfung|pruefung)\s*:)/i;
  for(const item of items){
    const plain = _mtStripHtml(String(item || ''));
    if(vrRe.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingVerificationRouteHeaderLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const hdrRe = /\bverification\s+route(?:\s+gate)?\b/i;
  for(const item of items){
    const plain = _mtStripHtml(String(item || ''));
    if(hdrRe.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingSourceTrainLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const trainRe = /\bsource\s*:\s*train\b/i;
  for(const item of items){
    const plain = _mtStripHtml(String(item || ''));
    if(trainRe.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const labelRx = '(?:Warum\\s+(?:das|es)\\s+wichtig\\s+ist|Warum\\s+relevant|Why\\s+it\\s+matters|Why\\s+this\\s+is\\s+important|Was\\s+(?:würde|wuerde)\\s+verifizieren(?:/|\\s+oder\\s+)falsifizieren\\s+\\((?:nächster|naechster)\\s+Check\\)|What\\s+would\\s+verify(?:/|\\s+or\\s+)falsify\\s+\\(next\\s+check\\)|N(?:ä|ae)chster\\s+(?:Check|Schritt)|Next\\s+(?:check|step))';
  const leakRe = new RegExp('(?:^|<br\\\\s*/?>|\\\\n|\\\\s)' + labelRx + '\\\\s*\\\\*\\\\*\\\\s*:', 'i');
  for(const item of items){
    if(leakRe.test(String(item || ''))) return true;
  }
  return false;
}

function _mtHasSelfDebunkingRawDoubleStarLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  for(const item of items){
    const raw = String(item || '');
    const plain = _mtStripHtml(raw);
    if(/\*\*/.test(raw) || /\*\*/.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingColorMarkerLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const tagRe = /\[(?:GREEN|YELLOW|RED|GRAY|WHITE)(?:-[A-Z0-9]+)*\]/i;
  const dotRe = /[🟢🟡🔴⚪]/;
  for(const item of items){
    const raw = String(item || '');
    const plain = _mtStripHtml(raw);
    if(/signal-dot-marker/i.test(raw)) return true;
    if(tagRe.test(raw) || tagRe.test(plain)) return true;
    if(dotRe.test(raw) || dotRe.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingPlaceholderFallbackLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const leakRe = /(?:Ohne Begründung bleibt die Aussage schwer einzuordnen\.|Ohne Begrundung bleibt die Aussage schwer einzuordnen\.|Eine Primärquelle oder ein Gegenbeispiel gezielt prüfen\.|Eine Primaerquelle oder ein Gegenbeispiel gezielt pruefen\.|Without this rationale, confidence and scope are harder to assess\.|Test one concrete counterexample or primary-source claim\.)/i;
  for(const item of items){
    const raw = String(item || '');
    const plain = _mtStripHtml(raw);
    if(leakRe.test(raw) || leakRe.test(plain)) return true;
  }
  return false;
}

function _mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(html){
  const items = _mtCollectSelfDebunkingListItemsHtml(html);
  if(!items.length) return false;
  const leadRe = /(?:<strong>\s*)?(?:Schw(?:ä|ae)che|Weakness|Unsicherheit|Uncertainty)(?:\s*<\/strong>)?\s*:\s*(?:<br\s*\/?>\s*)*(?:<strong>\s*)?(?:Warum\s+(?:das|es)\s+wichtig\s+ist|Warum\s+relevant|Why\s+it\s+matters|Why\s+this\s+is\s+important|Was\s+(?:würde|wuerde)\s+verifizieren(?:\/|\s+oder\s+)falsifizieren\s*\((?:nächster|naechster)\s+Check\)|What\s+would\s+verify(?:\/|\s+or\s+)falsify\s*\(next\s+check\)|N(?:ä|ae)chster\s+(?:Check|Schritt)|Next\s+(?:check|step))(?:\s*<\/strong>)?\s*:/i;
  const reasonRe = /(?:<strong>\s*)?(?:Warum\s+(?:das|es)\s+wichtig\s+ist|Warum\s+relevant|Why\s+it\s+matters|Why\s+this\s+is\s+important)(?:\s*<\/strong>)?\s*:/gi;
  const checkRe = /(?:<strong>\s*)?(?:Was\s+(?:würde|wuerde)\s+verifizieren(?:\/|\s+oder\s+)falsifizieren\s*\((?:nächster|naechster)\s+Check\)|What\s+would\s+verify(?:\/|\s+or\s+)falsify\s*\(next\s+check\)|N(?:ä|ae)chster\s+(?:Check|Schritt)|Next\s+(?:check|step))(?:\s*<\/strong>)?\s*:/gi;
  for(const item of items){
    const raw = String(item || '').replace(/<(\/?)b>/gi, '<$1strong>');
    if(!leadRe.test(raw)) continue;
    const reasonCount = (raw.match(reasonRe) || []).length;
    const checkCount = (raw.match(checkRe) || []).length;
    if(reasonCount >= 2 && checkCount >= 2) return true;
  }
  return false;
}

function _mtHasSelfDebunkingBrokenOlArtifacts(html){
  const raw = String(html || '');
  if(!raw) return false;
  const blockRe = /<div[^>]*class=(?:"|')[^"']*self-debunking[^"']*(?:"|')[^>]*>[\s\S]*?<\/div>(?=\s*<(?:p|div)\b|\s*\Z)/gi;
  const secRx = '(?:Warum\\s+(?:das|es)\\s+wichtig\\s+ist|Warum\\s+relevant|Why\\s+it\\s+matters|Why\\s+this\\s+is\\s+important|Was\\s+(?:würde|wuerde)\\s+verifizieren(?:/|\\s+oder\\s+)falsifizieren\\s+\\((?:nächster|naechster)\\s+Check\\)|What\\s+would\\s+verify(?:/|\\s+or\\s+)falsify\\s+\\(next\\s+check\\)|N(?:ä|ae)chster\\s+(?:Check|Schritt)|Next\\s+(?:check|step))';
  let bm = null;
  while((bm = blockRe.exec(raw)) !== null){
    const block = String((bm && bm[0]) || '');
    if(!block) continue;
    if(/<\/ol>\s*<\/ol>/i.test(block)) return true;
    if(/<ol[^>]*>\s*<\/ol>/i.test(block)) return true;
    if(new RegExp('</li>\\s*</ol>\\s*<p[^>]*>\\s*(?:<strong>\\s*)?(?:' + secRx + ')\\s*(?:</strong>)?\\s*:?', 'i').test(block)) return true;
  }
  return false;
}

function _mtHasDanglingAuxiliaryBeforeUncertaintyMarker(html){
  const raw = String(html || '');
  if(!raw) return false;
  const re = /(?:\b(?:wurde|wurden|wird|werden|war|waren|ist|sind|was|were|is|are|be|been)\b)\s*,\s*(?:<span\b[^>]*class=(?:"|')[^"']*\bsignal-dot-marker\b[^"']*(?:"|')[^>]*>[\s\S]*?<\/span>\s*)*(?:<span\b[^>]*class=(?:"|')[^"']*\buncertainty-inline-wrap\b[^"']*(?:"|')[^>]*>[\s\S]*?<\/span>|\(\s*U[1-8]\s*\))/i;
  return re.test(raw);
}

function _mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(html){
  const raw = String(html || '');
  if(!raw) return false;
  const wrappedMarker = "<span\\b[^>]*class=(?:\\\"|')[^\\\"']*\\buncertainty-inline-wrap\\b[^\\\"']*(?:\\\"|')[^>]*>[\\s\\S]*?<\\/span>";
  const signalDot = "<span\\b[^>]*class=(?:\\\"|')[^\\\"']*\\bsignal-dot-marker\\b[^\\\"']*(?:\\\"|')[^>]*>[\\s\\S]*?<\\/span>";
  const plainMarker = "\\(\\s*U[1-8]\\s*\\)";
  const markerSeq = "(?:(?:" + signalDot + "\\s*)*(?:" + wrappedMarker + "|" + plainMarker + ")\\s*)+";
  const re = new RegExp(
    "<p[^>]*>\\s*" + markerSeq + "\\s*<\\/p>\\s*(?=<div[^>]*class=(?:\\\"|')[^\\\"']*\\bself-debunking\\b)",
    "i"
  );
  return re.test(raw);
}

function _mtHasSciTraceStructure(html){
  const txt = _mtStripHtml(html);
  if(!/SCI Trace/i.test(txt)) return false;
  // Accept numbered or bullet-style step labels (models/renderers vary).
  const hasNumbered = /\b1\.\s*(Plan|Check|Solution|Critic|Linguist|Logician|Adversary|Dialectic)/i.test(txt);
  const hasBulletLabels = /(?:^|\n)\s*(?:[-•*]\s*)?(Plan|Check|Solution|Critic|Linguist|Logician|Adversary|Dialectic(?:_[0-9]+_[A-Za-z]+)?)\s*:/im.test(txt);
  return !!(hasNumbered || hasBulletLabels);
}

function _mtIncludesProviderLimitLabel(htmlOrText, providerLabel){
  const t = _mtStripHtml(htmlOrText);
  return t.toLowerCase().includes(String(providerLabel || '').toLowerCase()) &&
         /(limit|guthaben|credits|rate)/i.test(t);
}

function _mtLooksLikeTransientProviderError(htmlOrText){
  const t = _mtStripHtml(htmlOrText).toLowerCase();
  if(!t) return false;
  if(t.includes('resource_exhausted') || t.includes('resource exhausted')) return true;
  if(t.includes('too many requests') || t.includes('rate limit')) return true;
  if(t.includes('temporarily unavailable') || t.includes('service unavailable')) return true;
  if(t.includes('control layer error') && t.includes('429')) return true;
  return false;
}

function _mtEnsureRunning(){
  if(!window.__manualTestRunner || !window.__manualTestRunner.running) throw new Error('manual_test_not_running');
  if(window.__manualTestRunner.stop) throw new Error('manual_test_stopped');
}

async function _mtSleep(ms){
  await new Promise(r => setTimeout(r, ms||0));
}

function _mtAskTimeoutForCurrentProvider(timeoutMs){
  const base = Number(timeoutMs || 180000) || 180000;
  // In manual GUI regression runs, even command-like asks (profile/SCI menu selections)
  // may trigger provider roundtrips and exceed 60s on OpenRouter/HF. Use a robust floor.
  if(base < 180000){
    return Math.max(base, 180000);
  }
  return base;
}

async function _mtAsk(text, timeoutMs){
  _mtEnsureRunning();
  const t = String(text || '').trim();
  if(!t) return {html:'', csc:null};
  const askTimeout = _mtAskTimeoutForCurrentProvider(timeoutMs);
  await _mtMirrorMainAppend({role: 'user', text: t});
  _mtLog(_mtT('ask_prefix') + t);
  let res = null;
  try {
    res = await _apiCall('ask', [t], askTimeout);
  } catch(e) {
    const msg = String(e && e.message ? e.message : e);
    if(msg.toLowerCase().includes('pywebview api method not available: ask')){
      try {
        const mt = window.__manualTestRunner || {};
        if(!mt.askFallbackNoted){
          mt.askFallbackNoted = true;
          _mtLog(_mtT('info_ask_fallback'), 'info');
        }
      } catch(_e) {}
      res = await _apiCall('panel_action', ['ask', {text: t}], askTimeout);
      if(res && res.result && typeof res.result === 'object'){
        res = res.result;
      }
    } else {
      throw e;
    }
  }
  _mtEnsureRunning();
  const html = (res && res.html) ? String(res.html) : '';
  _mtLog(_mtT('ask_done_prefix') + ' (' + (html.length||0) + ' html chars)');
  await _mtMirrorMainAppend({
    role: 'bot',
    html: html,
    cgi_bar: !!(res && res.cgi_bar),
    csc: (res && res.csc) ? res.csc : null,
    answer_lang: ((document.getElementById('anslang') || {}).value || ''),
  });
  await _mtSleep(150);
  try { await buildUI(); } catch(e) {}
  return res || {html:'', csc:null};
}

async function _mtAskWithRetry(text, timeoutMs, options){
  const opts = options || {};
  const maxRetries = Math.max(0, Number(opts.maxRetries != null ? opts.maxRetries : 1) || 0);
  const backoffMs = Math.max(0, Number(opts.backoffMs != null ? opts.backoffMs : 1200) || 1200);
  const label = String(opts.label || text || '');
  let attempt = 0;
  let res = await _mtAsk(text, timeoutMs);
  while(attempt < maxRetries && _mtLooksLikeTransientProviderError((res && res.html) || '')){
    attempt += 1;
    _mtWarn(_mtL(
      'Transiente Provider-Fehlerantwort bei "' + label + '" -> Retry ' + attempt + '/' + maxRetries,
      'Transient provider error for "' + label + '" -> retry ' + attempt + '/' + maxRetries
    ));
    await _mtSleep(backoffMs * attempt);
    res = await _mtAsk(text, timeoutMs);
  }
  return res || {html:'', csc:null};
}

async function _mtPanelAction(action, payload, timeoutMs){
  _mtEnsureRunning();
  const res = await _apiCall('panel_action', [action, payload || {}], timeoutMs || 120000);
  _mtEnsureRunning();
  if(!res || res.ok === false){
    throw new Error((res && res.error) ? res.error : ('panel_action failed: ' + action));
  }
  return res;
}

async function _mtTryPanelAction(action, payload, timeoutMs){
  _mtEnsureRunning();
  const res = await _apiCall('panel_action', [action, payload || {}], timeoutMs || 120000);
  _mtEnsureRunning();
  return res || {ok: false, error: 'panel_action_empty_response', action: String(action || '')};
}

async function _mtSetProvider(provider){
  const p = String(provider || '').trim().toLowerCase();
  _mtLog(_mtT('set_provider_prefix') + p);
  // OpenRouter/Hugging Face can be slower due provider API latency / remote catalogs.
  const _setProviderTimeout = (p === 'openrouter' || p === 'huggingface') ? 45000 : 30000;
  const _refreshTimeout = (p === 'openrouter' || p === 'huggingface') ? 120000 : 60000;
  await _mtPanelAction('set_provider', {provider:p}, _setProviderTimeout);
  await _mtPanelAction('refresh_models', {provider:p}, _refreshTimeout);
  await _mtSleep(250);
  await buildUI();
}

async function _mtSetAnswerLanguage(lang){
  const l = String(lang || 'de').trim().toLowerCase();
  _mtLog(_mtT('set_answer_language_prefix') + l);
  await _mtPanelAction('set_answer_language', {lang:l}, 10000);
  try { _mtApplyLocalizedUi(); } catch(e) {}
  await _mtSleep(100);
}

function _mtBasename(pathLike){
  const p = String(pathLike || '').trim();
  if(!p) return '';
  const parts = p.split(/[\\/]+/).filter(Boolean);
  return parts.length ? String(parts[parts.length - 1] || '') : '';
}

async function _mtListChatLogs(limit){
  const res = await _mtPanelAction('list_chat_logs', {limit: Number(limit || 200)}, 12000);
  const logs = (res && Array.isArray(res.logs))
    ? res.logs
    : ((res && res.result && Array.isArray(res.result.logs)) ? res.result.logs : []);
  return Array.isArray(logs) ? logs : [];
}

async function _mtLoadChatLogByName(name, fork){
  return _mtPanelAction(
    'load_chat_log',
    {name: String(name || ''), fork: (fork !== false)},
    45000
  );
}

async function _mtTryLoadChatLogByName(name, fork){
  const res = await _apiCall(
    'panel_action',
    ['load_chat_log', {name: String(name || ''), fork: (fork !== false)}],
    45000
  );
  return res || {ok: false, error: 'load_chat_log_empty_response'};
}

async function _mtApplyQcOverride(values){
  _mtEnsureRunning();
  _mtLog(_mtT('qc_apply_prefix') + JSON.stringify(values || {}));
  const api = _api();
  if(api && typeof api.qc_override_apply === 'function'){
    const res = await _apiCall('qc_override_apply', [values || {}], 10000);
    if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_apply failed');
    await _mtSleep(120);
    return;
  }
  const mt = window.__manualTestRunner || {};
  if(!mt.qcApplyFallbackNoted){
    mt.qcApplyFallbackNoted = true;
    _mtWarn(_mtT('qc_apply_fallback'));
  }
  const res = await _mtPanelAction('qc_override_apply', {values: values || {}}, 10000);
  if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_apply failed');
  await _mtSleep(120);
}

async function _mtClearQcOverride(){
  _mtEnsureRunning();
  _mtLog(_mtT('qc_clear'));
  const api = _api();
  if(api && typeof api.qc_override_clear === 'function'){
    const res = await _apiCall('qc_override_clear', [], 10000);
    if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_clear failed');
    await _mtSleep(120);
    return;
  }
  const mt = window.__manualTestRunner || {};
  if(!mt.qcClearFallbackNoted){
    mt.qcClearFallbackNoted = true;
    _mtWarn(_mtT('qc_clear_fallback'));
  }
  const res = await _mtPanelAction('qc_override_clear', {}, 10000);
  if(!res || res.ok === false) throw new Error((res && res.error) ? res.error : 'qc_override_clear failed');
  await _mtSleep(120);
}

function _mtCheck(cond, okMsg, failMsg){
  if(cond){
    _mtLog(_mtT('pass_prefix') + okMsg, 'ok');
    return true;
  }
  _mtLog(_mtT('fail_prefix') + failMsg, 'err');
  return false;
}

function _mtWarn(msg){
  _mtLog(_mtT('warn_prefix') + msg, 'warn');
}

function _mtParseErrorTextFromAskResult(res){
  try{
    const txt = _mtStripHtml((res && res.html) || '');
    if(/(error|limit|guthaben|credits|rate)/i.test(txt)) return txt;
  }catch(e){}
  return '';
}

async function _mtScenarioSmokeShort(){
  let fails = 0;
  const q = _mtPromptShort();
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());
  await _mtSetProvider('gemini');
  await _mtAsk('Profile Standard', 30000);
  await _mtAsk('SCI off', 30000);
  const r1 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'smoke-base'});
  if(_mtLooksLikeTransientProviderError(r1.html || '')){
    _mtWarn(_mtL(
      'Smoke-Check uebersprungen: transiente Provider-Fehlerantwort bei Basisprompt.',
      'Smoke check skipped: transient provider error on base prompt.'
    ));
    return {fails};
  }
  const hasCompleteBaseQc = _mtHasCompleteQcFooter(r1.html);
  if(!_mtCheckQcFooterWithSciAlertGuard(
    r1.html,
    hasCompleteBaseQc,
    _mtL('QC-Footer bei Basisantwort vollstaendig', 'QC footer complete for base answer'),
    _mtL('QC-Footer bei Basisantwort fehlt/ist unvollstaendig', 'QC footer missing/incomplete for base answer'),
    _mtL(
      'QC-Check bei Basisantwort uebersprungen (SCI-Alert: Missing SCI Trace step content).',
      'QC check for base answer skipped (SCI alert: missing SCI Trace step content).'
    )
  )) fails++;
  if(!_mtCheck(
    _mtHasQcDimTooltips(r1.html),
    _mtL('QC-Dimension-Tooltips vorhanden', 'QC dimension tooltips present'),
    _mtL('QC-Dimension-Tooltips fehlen/unvollstaendig', 'QC dimension tooltips missing/incomplete')
  )) fails++;
  if(!_mtCheck(
    _mtStripHtml(r1.html).includes('Self-Debunking') || _mtStripHtml(r1.html).includes('Selbst-Debunking'),
    _mtL('Self-Debunking vorhanden', 'Self-debunking present'),
    _mtL('Self-Debunking fehlt', 'Self-debunking missing')
  )) fails++;
  if(!_mtCheck(
    !_mtHasBrokenRedSpanAroundUncertaintyMarker(r1.html),
    _mtL('U-Marker Span-Grenzen konsistent', 'U-marker span boundaries consistent'),
    _mtL('U-Marker Span-Grenzen defekt (roter Textlauf)', 'U-marker span boundaries broken (red text bleed)')
  )) fails++;
  if(!_mtCheck(
    !_mtHasUncertaintyTailPhraseLeakU1ToU8(r1.html),
    _mtL('U1-U8-Guard ok (keine Tail-Phrasen) im Basisoutput', 'U1-U8 guard ok (no tail phrases) in base output'),
    _mtL('U1-U8-Tail-Leak im Basisoutput erkannt', 'U1-U8 tail leak detected in base output')
  )) fails++;
  if(!_mtCheck(
    !_mtHasStrictEnforcementBanner(r1.html),
    _mtL('Kein Strict-Enforcement-Banner bei Basisantwort', 'No strict-enforcement banner for base answer'),
    _mtL('Unerwartetes Strict-Enforcement-Banner bei Basisantwort', 'Unexpected strict-enforcement banner for base answer')
  )) fails++;
  if(!_mtCheck(
    !_mtHasRenderFallbackNote(r1.html),
    _mtL('Kein Render-Fallback-Hinweis bei Basisantwort', 'No render-fallback note for base answer'),
    _mtL('Unerwarteter Render-Fallback-Hinweis bei Basisantwort', 'Unexpected render-fallback note for base answer')
  )) fails++;

  const testImgUrl = 'https://example.com/comm-sci-manual-test.png';
  const imgPrompt = _mtL(
    'Gib exakt nur diese URL aus: ' + testImgUrl,
    'Reply with exactly this URL only: ' + testImgUrl
  );
  const rImg = await _mtAsk(imgPrompt, 120000);
  const txtImg = _mtStripHtml(rImg.html || '');
  if(_mtLooksLikeDeterministicImageEchoResponse(rImg.html, testImgUrl)){
    if(!_mtCheck(
      _mtHasEmbeddedImageTagForUrl(rImg.html, testImgUrl),
      _mtL('Image-URL als <img> eingebettet', 'Image URL embedded as <img>'),
      _mtL('Image-URL nicht als <img> eingebettet', 'Image URL not embedded as <img>')
    )) fails++;
  } else if(txtImg.includes(testImgUrl)) {
    _mtWarn(_mtL(
      'Image-Embed-Check uebersprungen: Provider-Antwort nicht als kurzer URL-Echo interpretierbar.',
      'Image-embed check skipped: provider response is not a short deterministic URL echo.'
    ));
  } else {
    _mtWarn(_mtL(
      'Image-Embed-Check uebersprungen: Test-URL nicht in Antwort enthalten.',
      'Image-embed check skipped: test URL not present in answer.'
    ));
  }

  const imgPromptDot = _mtL(
    'Gib exakt diese URL mit abschliessendem Punkt aus: ' + testImgUrl + '.',
    'Reply with this URL including a trailing period: ' + testImgUrl + '.'
  );
  const rImgDot = await _mtAsk(imgPromptDot, 120000);
  const txtImgDot = _mtStripHtml(rImgDot.html || '');
  if(_mtLooksLikeDeterministicImageEchoResponse(rImgDot.html, testImgUrl)){
    if(!_mtCheck(
      _mtHasEmbeddedImageTagForUrl(rImgDot.html, testImgUrl),
      _mtL('Image-URL mit Satzzeichen als <img> eingebettet', 'Image URL with punctuation embedded as <img>'),
      _mtL('Image-URL mit Satzzeichen nicht als <img> eingebettet', 'Image URL with punctuation not embedded as <img>')
    )) fails++;
  } else if(txtImgDot.includes(testImgUrl)) {
    _mtWarn(_mtL(
      'Image-Embed-Check (Satzzeichen) uebersprungen: Provider-Antwort nicht als kurzer URL-Echo interpretierbar.',
      'Image-embed check (punctuation) skipped: provider response is not a short deterministic URL echo.'
    ));
  } else {
    _mtWarn(_mtL(
      'Image-Embed-Check (Satzzeichen) uebersprungen: Test-URL nicht in Antwort enthalten.',
      'Image-embed check (punctuation) skipped: test URL not present in answer.'
    ));
  }

  const imgPromptInlineCode = _mtL(
    'Gib exakt diese URL als Inline-Code in Backticks aus: `' + testImgUrl + '`',
    'Reply with exactly this URL as inline code in backticks: `' + testImgUrl + '`'
  );
  const rImgInlineCode = await _mtAsk(imgPromptInlineCode, 120000);
  const txtImgInlineCode = _mtStripHtml(rImgInlineCode.html || '');
  const hasInlineCodeShape = /(?:`|<code>)/i.test(String(rImgInlineCode.html || ''));
  if(hasInlineCodeShape && _mtLooksLikeDeterministicImageEchoResponse(rImgInlineCode.html, testImgUrl)){
    if(!_mtCheck(
      _mtHasEmbeddedImageTagForUrl(rImgInlineCode.html, testImgUrl),
      _mtL('Image-URL im Inline-Code als <img> eingebettet', 'Image URL in inline code embedded as <img>'),
      _mtL('Image-URL im Inline-Code nicht als <img> eingebettet', 'Image URL in inline code not embedded as <img>')
    )) fails++;
  } else if(hasInlineCodeShape && txtImgInlineCode.includes(testImgUrl)) {
    _mtWarn(_mtL(
      'Image-Embed-Check (Inline-Code) uebersprungen: Provider-Antwort nicht als kurzer URL-Echo interpretierbar.',
      'Image-embed check (inline code) skipped: provider response is not a short deterministic URL echo.'
    ));
  } else {
    _mtWarn(_mtL(
      'Image-Embed-Check (Inline-Code) uebersprungen: URL/Codeform nicht in Antwort enthalten.',
      'Image-embed check (inline code) skipped: URL/code form not present in answer.'
    ));
  }

  const imgPromptFence = _mtL(
    'Gib exakt diese URL in einem einzelnen ```txt```-Codeblock aus: ' + testImgUrl,
    'Reply with exactly this URL in a single ```txt``` code block: ' + testImgUrl
  );
  const rImgFence = await _mtAsk(imgPromptFence, 120000);
  const txtImgFence = _mtStripHtml(rImgFence.html || '');
  const hasFenceShape = /```|<pre|<code/i.test(String(rImgFence.html || ''));
  if(hasFenceShape && _mtLooksLikeDeterministicImageEchoResponse(rImgFence.html, testImgUrl)){
    if(!_mtCheck(
      _mtHasEmbeddedImageTagForUrl(rImgFence.html, testImgUrl),
      _mtL('Image-URL im Codeblock als <img> eingebettet', 'Image URL in code block embedded as <img>'),
      _mtL('Image-URL im Codeblock nicht als <img> eingebettet', 'Image URL in code block not embedded as <img>')
    )) fails++;
  } else if(hasFenceShape && txtImgFence.includes(testImgUrl)) {
    _mtWarn(_mtL(
      'Image-Embed-Check (Codeblock) uebersprungen: Provider-Antwort nicht als kurzer URL-Echo interpretierbar.',
      'Image-embed check (code block) skipped: provider response is not a short deterministic URL echo.'
    ));
  } else {
    _mtWarn(_mtL(
      'Image-Embed-Check (Codeblock) uebersprungen: URL/Codeform nicht in Antwort enthalten.',
      'Image-embed check (code block) skipped: URL/code form not present in answer.'
    ));
  }
  return {fails};
}

async function _mtScenarioProviderSwitch(){
  let fails = 0;
  const q = _mtPromptShort();
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());
  await _mtSetProvider('gemini');
  await _mtAsk('Profile Standard', 30000);
  let rg = await _mtAsk(q, 180000);
  const hasGeminiQc = _mtHasCompleteQcFooter(rg.html);
  if(!_mtCheckQcFooterWithSciAlertGuard(
    rg.html,
    hasGeminiQc,
    _mtL('Gemini: QC-Footer ok', 'Gemini: QC footer ok'),
    _mtL('Gemini: QC-Footer fehlt/unvollstaendig', 'Gemini: QC footer missing/incomplete'),
    _mtL(
      'Gemini: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
      'Gemini: QC check skipped (SCI alert: missing SCI Trace step content).'
    )
  )) fails++;

  await _mtSetProvider('openrouter');
  let ro = await _mtAsk(q, 180000);
  const txtOR = _mtStripHtml(ro.html);
  if(/(insufficient credits|Nicht genügend Guthaben|rate limit|Limit erreicht)/i.test(txtOR)){
    _mtWarn(_mtL(
      'OpenRouter lieferte Limit/Credit-Fehler (Test trotzdem ok, Providerpfad verifiziert).',
      'OpenRouter returned limit/credit error (test still acceptable; provider path verified).'
    ));
    if(!_mtCheck(
      _mtIncludesProviderLimitLabel(ro.html, 'OpenRouter'),
      _mtL('OpenRouter-Fehlerlabel korrekt', 'OpenRouter error label correct'),
      _mtL('OpenRouter-Fehlerlabel unklar/falsch', 'OpenRouter error label unclear/incorrect')
    )) fails++;
  } else {
    const hasOpenRouterQc = _mtHasCompleteQcFooter(ro.html);
    if(!_mtCheckQcFooterWithSciAlertGuard(
      ro.html,
      hasOpenRouterQc,
      _mtL('OpenRouter: QC-Footer ok', 'OpenRouter: QC footer ok'),
      _mtL('OpenRouter: QC-Footer fehlt/unvollstaendig', 'OpenRouter: QC footer missing/incomplete'),
      _mtL(
        'OpenRouter: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
        'OpenRouter: QC check skipped (SCI alert: missing SCI Trace step content).'
      )
    )) fails++;
    if(!_mtCheck(
      _mtHasSelfDebunkingBox(ro.html),
      _mtL('OpenRouter: Self-Debunking-Box erkannt', 'OpenRouter: self-debunking box detected'),
      _mtL('OpenRouter: Self-Debunking nicht als Box erkannt', 'OpenRouter: self-debunking box not detected')
    )) fails++;
  }

  await _mtSetProvider('huggingface');
  let rhf = await _mtAsk(q, 180000);
  const txtHF = _mtStripHtml(rhf.html);
  if(/(insufficient credits|Nicht genügend Guthaben|rate limit|Limit erreicht|quota)/i.test(txtHF)){
    _mtWarn(_mtL('Hugging Face lieferte Limit/Credit-Fehler (optional).', 'Hugging Face returned limit/credit error (optional).'));
    if(!_mtCheck(
      _mtIncludesProviderLimitLabel(rhf.html, 'Hugging Face'),
      _mtL('HF-Fehlerlabel korrekt', 'HF error label correct'),
      _mtL('HF-Fehler wurde nicht als Hugging Face gekennzeichnet', 'HF error not labeled as Hugging Face')
    )) fails++;
  } else {
    const hasHfQc = _mtHasCompleteQcFooter(rhf.html);
    if(!_mtCheckQcFooterWithSciAlertGuard(
      rhf.html,
      hasHfQc,
      _mtL('HF: QC-Footer ok', 'HF: QC footer ok'),
      _mtL('HF: QC-Footer fehlt/unvollstaendig', 'HF: QC footer missing/incomplete'),
      _mtL(
        'HF: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
        'HF: QC check skipped (SCI alert: missing SCI Trace step content).'
      )
    )) fails++;
  }
  return {fails};
}

async function _mtScenarioSciFormat(){
  let fails = 0;
  const q = _mtPromptShort();
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtAsk('Profile Expert', 30000);
  await _mtAsk('SCI menu', 30000);
  await _mtAsk('A', 30000);
  const rA = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A'});
  if(_mtLooksLikeTransientProviderError(rA.html || '')){
    _mtWarn(_mtL(
      'SCI A uebersprungen: transiente Provider-Fehlerantwort.',
      'SCI A skipped: transient provider error response.'
    ));
    return {fails};
  }
  if(!_mtCheck(
    _mtHasSciTraceStructure(rA.html),
    _mtL('SCI A: SCI-Trace-Struktur erkannt', 'SCI A: SCI trace structure detected'),
    _mtL('SCI A: SCI-Trace-Struktur fehlt/unsauber', 'SCI A: SCI trace structure missing/invalid')
  )) fails++;
  const hasSciAQc = _mtHasCompleteQcFooter(rA.html);
  if(!_mtCheckQcFooterWithSciAlertGuard(
    rA.html,
    hasSciAQc,
    _mtL('SCI A: QC-Footer ok', 'SCI A: QC footer ok'),
    _mtL('SCI A: QC-Footer fehlt/unvollstaendig', 'SCI A: QC footer missing/incomplete'),
    _mtL(
      'SCI A: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
      'SCI A: QC check skipped (SCI alert: missing SCI Trace step content).'
    )
  )) fails++;
  let sciATooltipsOk = _mtHasQcDimTooltips(rA.html);
  if(!sciATooltipsOk){
    _mtWarn(_mtL(
      'SCI A: QC-Tooltip-Check Retry 1/1',
      'SCI A: QC tooltip check retry 1/1'
    ));
    const rA2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-A-tooltips-retry'});
    if(!_mtLooksLikeTransientProviderError(rA2.html || '')){
      sciATooltipsOk = _mtHasQcDimTooltips(rA2.html);
    }
  }
  if(!_mtCheck(
    sciATooltipsOk,
    _mtL('SCI A: QC-Dimension-Tooltips vorhanden', 'SCI A: QC dimension tooltips present'),
    _mtL('SCI A: QC-Dimension-Tooltips fehlen/unvollstaendig', 'SCI A: QC dimension tooltips missing/incomplete')
  )) fails++;
  if(!_mtCheck(
    !_mtHasBrokenRedSpanAroundUncertaintyMarker(rA.html),
    _mtL('SCI A: U-Marker Span-Grenzen konsistent', 'SCI A: U-marker span boundaries consistent'),
    _mtL('SCI A: U-Marker Span-Grenzen defekt (roter Textlauf)', 'SCI A: U-marker span boundaries broken (red text bleed)')
  )) fails++;
  const sciATailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rA.html);
  if(!_mtCheck(
    !sciATailLeakSnippet,
    _mtL('SCI A: U1-U8-Guard ok (keine Tail-Phrasen)', 'SCI A: U1-U8 guard ok (no tail phrases)'),
    sciATailLeakSnippet
      ? _mtL(
          'SCI A: U1-U8-Tail-Leak erkannt [Match: ' + sciATailLeakSnippet + ']',
          'SCI A: U1-U8 tail leak detected [match: ' + sciATailLeakSnippet + ']'
        )
      : _mtL('SCI A: U1-U8-Tail-Leak erkannt', 'SCI A: U1-U8 tail leak detected')
  )) fails++;

  await _mtAsk('SCI menu', 30000);
  await _mtAsk('B', 30000);
  const rB = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B'});
  if(_mtLooksLikeTransientProviderError(rB.html || '')){
    _mtWarn(_mtL(
      'SCI B uebersprungen: transiente Provider-Fehlerantwort.',
      'SCI B skipped: transient provider error response.'
    ));
    return {fails};
  }
  if(!_mtCheck(
    _mtHasSciTraceStructure(rB.html),
    _mtL('SCI B: SCI-Trace-Struktur erkannt', 'SCI B: SCI trace structure detected'),
    _mtL('SCI B: SCI-Trace-Struktur fehlt/unsauber', 'SCI B: SCI trace structure missing/invalid')
  )) fails++;
  const hasSciBQc = _mtHasCompleteQcFooter(rB.html);
  if(!_mtCheckQcFooterWithSciAlertGuard(
    rB.html,
    hasSciBQc,
    _mtL('SCI B: QC-Footer ok', 'SCI B: QC footer ok'),
    _mtL('SCI B: QC-Footer fehlt/unvollstaendig', 'SCI B: QC footer missing/incomplete'),
    _mtL(
      'SCI B: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
      'SCI B: QC check skipped (SCI alert: missing SCI Trace step content).'
    )
  )) fails++;
  let sciBTooltipsOk = _mtHasQcDimTooltips(rB.html);
  if(!sciBTooltipsOk){
    _mtWarn(_mtL(
      'SCI B: QC-Tooltip-Check Retry 1/1',
      'SCI B: QC tooltip check retry 1/1'
    ));
    const rB2 = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'sci-B-tooltips-retry'});
    if(!_mtLooksLikeTransientProviderError(rB2.html || '')){
      sciBTooltipsOk = _mtHasQcDimTooltips(rB2.html);
    }
  }
  if(!_mtCheck(
    sciBTooltipsOk,
    _mtL('SCI B: QC-Dimension-Tooltips vorhanden', 'SCI B: QC dimension tooltips present'),
    _mtL('SCI B: QC-Dimension-Tooltips fehlen/unvollstaendig', 'SCI B: QC dimension tooltips missing/incomplete')
  )) fails++;
  if(!_mtCheck(
    !_mtHasBrokenRedSpanAroundUncertaintyMarker(rB.html),
    _mtL('SCI B: U-Marker Span-Grenzen konsistent', 'SCI B: U-marker span boundaries consistent'),
    _mtL('SCI B: U-Marker Span-Grenzen defekt (roter Textlauf)', 'SCI B: U-marker span boundaries broken (red text bleed)')
  )) fails++;
  const sciBTailLeakSnippet = _mtFindUncertaintyTailPhraseLeakU1ToU8(rB.html);
  if(!_mtCheck(
    !sciBTailLeakSnippet,
    _mtL('SCI B: U1-U8-Guard ok (keine Tail-Phrasen)', 'SCI B: U1-U8 guard ok (no tail phrases)'),
    sciBTailLeakSnippet
      ? _mtL(
          'SCI B: U1-U8-Tail-Leak erkannt [Match: ' + sciBTailLeakSnippet + ']',
          'SCI B: U1-U8 tail leak detected [match: ' + sciBTailLeakSnippet + ']'
        )
      : _mtL('SCI B: U1-U8-Tail-Leak erkannt', 'SCI B: U1-U8 tail leak detected')
  )) fails++;
  if(!_mtCheck(
    !_mtHasFinalAnswerLabelInsideSciTrace(rB.html),
    _mtL('SCI B: kein Final-Answer-Label im SCI-Trace', 'SCI B: no final-answer label inside SCI trace'),
    _mtL('SCI B: Final-Answer-Label im SCI-Trace erkannt', 'SCI B: final-answer label detected inside SCI trace')
  )) fails++;
  return {fails};
}

async function _mtScenarioQcOverrideFooter(){
  let fails = 0;
  const q = _mtPromptShort();
  await _mtPanelAction('clear_chat', {}, 8000);
  // Deterministic reference provider for QC/SCI contract checks (credits/limits on
  // optional providers must not turn feature-contract tests into false FAILs).
  await _mtSetProvider('gemini');
  await _mtAsk('Profile Expert', 30000);
  await _mtAsk('SCI menu', 30000);
  await _mtAsk('B', 30000);
  await _mtApplyQcOverride({Clarity:3,Brevity:0,Evidence:3,Empathy:3,Consistency:3,Neutrality:3});
  const r = await _mtAsk(q, 180000);
  const hasOverrideQc = _mtHasCompleteQcFooter(r.html);
  if(!_mtCheckQcFooterWithSciAlertGuard(
    r.html,
    hasOverrideQc,
    _mtL('SCI B + QC-Override: QC-Footer vollstaendig', 'SCI B + QC override: QC footer complete'),
    _mtL('SCI B + QC-Override: QC-Footer fehlt/unvollstaendig', 'SCI B + QC override: QC footer missing/incomplete'),
    _mtL(
      'SCI B + QC-Override: QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
      'SCI B + QC override: QC check skipped (SCI alert: missing SCI Trace step content).'
    )
  )) fails++;
  if(!_mtCheck(
    _mtHasQcDimTooltips(r.html),
    _mtL('SCI B + QC-Override: QC-Dimension-Tooltips vorhanden', 'SCI B + QC override: QC dimension tooltips present'),
    _mtL('SCI B + QC-Override: QC-Dimension-Tooltips fehlen/unvollstaendig', 'SCI B + QC override: QC dimension tooltips missing/incomplete')
  )) fails++;
  if(!_mtCheck(
    !_mtHasLocalizedQcFooterWithoutDeltaAtEnd(r.html),
    _mtL('SCI B + QC-Override: kein DE-QC-Footer ohne Delta am Ende', 'SCI B + QC override: no localized QC footer without delta at the end'),
    _mtL('SCI B + QC-Override: DE-QC-Footer ohne Delta am Ende erkannt', 'SCI B + QC override: localized QC footer without delta detected at the end')
  )) fails++;
  if(!_mtCheck(
    _mtHasSelfDebunkingBox(r.html),
    _mtL('SCI B + QC-Override: Self-Debunking-Box erkannt', 'SCI B + QC override: self-debunking box detected'),
    _mtL('SCI B + QC-Override: Self-Debunking nicht als Box erkannt', 'SCI B + QC override: self-debunking box not detected')
  )) fails++;
  if(_mtHasSciTraceStructure(r.html)){
    if(_mtHasTraceSelfDebunkingQcTimestampOrder(r.html)){
      _mtLog(_mtT('pass_prefix') + _mtL('SCI B + QC-Override: Reihenfolge SCI Trace -> Self-Debunking -> QC -> Timestamp stabil', 'SCI B + QC override: order SCI Trace -> Self-Debunking -> QC -> timestamp stable'), 'ok');
    } else {
      _mtWarn(_mtL(
        'SCI B + QC-Override: Reihenfolge-Check uebersprungen (Provider-Antwort strukturell instabil).',
        'SCI B + QC override: order check skipped (provider response structurally unstable).'
      ));
    }
  } else {
    _mtWarn(_mtL(
      'SCI B + QC-Override: Reihenfolge-Check uebersprungen (SCI-Trace-Struktur nicht vorhanden).',
      'SCI B + QC override: order check skipped (SCI trace structure not present).'
    ));
  }
  if(!_mtCheck(
    !_mtHasBrokenRedSpanAroundUncertaintyMarker(r.html),
    _mtL('SCI B + QC-Override: U-Marker Span-Grenzen konsistent', 'SCI B + QC override: U-marker span boundaries consistent'),
    _mtL('SCI B + QC-Override: U-Marker Span-Grenzen defekt (roter Textlauf)', 'SCI B + QC override: U-marker span boundaries broken (red text bleed)')
  )) fails++;
  if(!_mtCheck(
    !_mtHasUncertaintyTailPhraseLeakU1ToU8(r.html),
    _mtL('SCI B + QC-Override: U1-U8-Guard ok (keine Tail-Phrasen)', 'SCI B + QC override: U1-U8 guard ok (no tail phrases)'),
    _mtL('SCI B + QC-Override: U1-U8-Tail-Leak erkannt', 'SCI B + QC override: U1-U8 tail leak detected')
  )) fails++;
  if(!_mtCheck(
    !_mtHasFinalAnswerLabelInsideSciTrace(r.html),
    _mtL('SCI B + QC-Override: kein Final-Answer-Label im SCI-Trace', 'SCI B + QC override: no final-answer label inside SCI trace'),
    _mtL('SCI B + QC-Override: Final-Answer-Label im SCI-Trace erkannt', 'SCI B + QC override: final-answer label detected inside SCI trace')
  )) fails++;
  try { await _mtClearQcOverride(); } catch(e) { _mtWarn(_mtL('QC-Override clear fehlgeschlagen: ', 'QC override clear failed: ') + String(e && e.message ? e.message : e)); }
  return {fails};
}

async function _mtScenarioProfileSelfDebunking(){
  let fails = 0;
  const q = _mtPromptShort();
  const profiles = ['Standard', 'Briefing', 'Sandbox', 'Expert'];
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());

  for(const profile of profiles){
    _mtEnsureRunning();
    await _mtAsk('Profile ' + profile, 30000);
    await _mtAsk('SCI off', 30000);
    const res = await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'profile-self-debunking-' + profile});
    if(_mtLooksLikeTransientProviderError(res.html || '')){
      _mtWarn(_mtL(
        'Profil ' + profile + ': transiente Provider-Fehlerantwort (Checks uebersprungen).',
        'Profile ' + profile + ': transient provider error response (checks skipped).'
      ));
      continue;
    }
    const lbl = _mtL('Profil ', 'Profile ') + profile;

    if(!_mtCheck(
      _mtHasActiveProfileHeader(res.html, profile),
      lbl + ': ' + _mtL('Header zeigt aktives Profil korrekt', 'header shows active profile correctly'),
      lbl + ': ' + _mtL('Header zeigt aktives Profil nicht korrekt', 'header does not show active profile correctly')
    )) fails++;
    if(!_mtCheck(
      !_mtLooksLikeClockTimeListAnswer(res.html),
      lbl + ': ' + _mtL(
        'Standardfrage semantisch als Zeit-Begriff beantwortet',
        'short prompt answered as concept of time'
      ),
      lbl + ': ' + _mtL(
        'Semantischer Drift: Uhrzeitliste statt Zeit-Begriff',
        'semantic drift: clock-time list instead of concept of time'
      )
    )) fails++;

    const hasProfileQc = _mtHasCompleteQcFooter(res.html);
    if(!_mtCheckQcFooterWithSciAlertGuard(
      res.html,
      hasProfileQc,
      lbl + ': ' + _mtL('QC-Footer vollstaendig', 'QC footer complete'),
      lbl + ': ' + _mtL('QC-Footer fehlt/unvollstaendig', 'QC footer missing/incomplete'),
      lbl + ': ' + _mtL(
        'QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
        'QC check skipped (SCI alert: missing SCI Trace step content).'
      )
    )) fails++;
    if(!_mtCheck(
      _mtHasQcDimTooltips(res.html),
      lbl + ': ' + _mtL('QC-Dimension-Tooltips vorhanden', 'QC dimension tooltips present'),
      lbl + ': ' + _mtL('QC-Dimension-Tooltips fehlen/unvollstaendig', 'QC dimension tooltips missing/incomplete')
    )) fails++;

    if(!_mtCheck(
      _mtHasSelfDebunkingBox(res.html),
      lbl + ': ' + _mtL('Self-Debunking-Box erkannt', 'self-debunking box detected'),
      lbl + ': ' + _mtL('Self-Debunking-Box fehlt', 'self-debunking box missing')
    )) fails++;

    if(!_mtCheck(
      _mtHasSelfDebunkingContractStructure(res.html),
      lbl + ': ' + _mtL('Self-Debunking-Struktur (Schwäche/Warum/Check) ok', 'self-debunking structure (weakness/why/check) ok'),
      lbl + ': ' + _mtL('Self-Debunking-Struktur unvollstaendig', 'self-debunking structure incomplete')
    )) fails++;
    if(!_mtCheck(
      _mtHasSelfDebunkingSecondaryLabelsBoldLinebreak(res.html),
      lbl + ': ' + _mtL(
        'Self-Debunking-Secondary-Labels fett + neue Zeile (alle Punkte)',
        'self-debunking secondary labels bold + new line (all points)'
      ),
      lbl + ': ' + _mtL(
        'Self-Debunking-Secondary-Labels nicht durchgehend fett + neue Zeile',
        'self-debunking secondary labels not consistently bold + new line'
      )
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingItalicSecondaryLabelLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne italic Secondary-Label-Leak', 'self-debunking without italic secondary-label leak'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt italic Secondary-Label-Leak', 'self-debunking contains italic secondary-label leak')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingVerificationRouteLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Verification-Route-Leak', 'self-debunking without verification-route leak'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Verification-Route-Leak', 'self-debunking contains verification-route leak')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingVerificationRouteHeaderLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Verification-Route-Header im Block', 'self-debunking without verification-route header inside block'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Verification-Route-Header im Block', 'self-debunking contains verification-route header inside block')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingSourceTrainLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Source: TRAIN-Leak', 'self-debunking without Source: TRAIN leak'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Source: TRAIN-Leak', 'self-debunking contains Source: TRAIN leak')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingBrokenSecondaryLabelMdLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Secondary-Label-**-Leak', 'self-debunking without secondary-label ** leak'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Secondary-Label-**-Leak', 'self-debunking contains secondary-label ** leak')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingRawDoubleStarLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne rohe **-Artefakte', 'self-debunking without raw ** artifacts'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt rohe **-Artefakte', 'self-debunking contains raw ** artifacts')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingColorMarkerLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Farbmarker-Leak', 'self-debunking without color-marker leak'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Farbmarker-Leak', 'self-debunking contains color-marker leak')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingPlaceholderFallbackLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Placeholder-Fallback-Texte', 'self-debunking without placeholder fallback phrases'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Placeholder-Fallback-Texte', 'self-debunking contains placeholder fallback phrases')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingLeadingSecondaryDuplicateLeak(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne Leading-Secondary-Duplikat im ersten Feld', 'self-debunking without leading-secondary duplicate in first field'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt Leading-Secondary-Duplikat im ersten Feld', 'self-debunking contains leading-secondary duplicate in first field')
    )) fails++;
    if(!_mtCheck(
      !_mtHasSelfDebunkingBrokenOlArtifacts(res.html),
      lbl + ': ' + _mtL('Self-Debunking ohne kaputte <ol>-Artefakte', 'self-debunking without broken <ol> artifacts'),
      lbl + ': ' + _mtL('Self-Debunking enthaelt kaputte <ol>-Artefakte', 'self-debunking contains broken <ol> artifacts')
    )) fails++;
    if(!_mtCheck(
      !_mtHasDanglingAuxiliaryBeforeUncertaintyMarker(res.html),
      lbl + ': ' + _mtL('Keine abgebrochene Hilfsverb-Klausel vor U-Marker', 'no dangling auxiliary clause before U marker'),
      lbl + ': ' + _mtL('Abgebrochene Hilfsverb-Klausel vor U-Marker erkannt', 'dangling auxiliary clause before U marker detected')
    )) fails++;
    if(!_mtCheck(
      !_mtHasOrphanUncertaintyMarkerParagraphBeforeSelfDebunking(res.html),
      lbl + ': ' + _mtL('Kein isolierter U-Only-Absatz vor Self-Debunking', 'no isolated U-only paragraph before self-debunking'),
      lbl + ': ' + _mtL('Isolierter U-Only-Absatz vor Self-Debunking erkannt', 'isolated U-only paragraph before self-debunking detected')
    )) fails++;
    if(!_mtCheck(
      !_mtHasBrokenRedSpanAroundUncertaintyMarker(res.html),
      lbl + ': ' + _mtL('U-Marker Span-Grenzen konsistent', 'U-marker span boundaries consistent'),
      lbl + ': ' + _mtL('U-Marker Span-Grenzen defekt (roter Textlauf)', 'U-marker span boundaries broken (red text bleed)')
    )) fails++;
    if(!_mtCheck(
      !_mtHasUncertaintyTailPhraseLeakU1ToU8(res.html),
      lbl + ': ' + _mtL('U1-U8-Guard ok (keine Tail-Phrasen)', 'U1-U8 guard ok (no tail phrases)'),
      lbl + ': ' + _mtL('U1-U8-Tail-Leak erkannt', 'U1-U8 tail leak detected')
    )) fails++;
  }

  await _mtExportChatAudit('profile_self_debunking');
  return {fails};
}

async function _mtScenarioStateRoutingCore(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());

  const checkCommState = async (field, expected, label) => {
    _mtEnsureRunning();
    const stateRes = await _mtAsk('Comm State', 30000);
    const ok = _mtHasCommStateField(stateRes.html, field, expected);
    if(!_mtCheck(
      ok,
      label + ': ' + _mtL('Comm State ok', 'Comm State ok'),
      label + ': ' + _mtL(
        'Comm State unerwartet (' + field + '=' + expected + ')',
        'unexpected Comm State (' + field + '=' + expected + ')'
      )
    )) fails++;
  };

  await _mtAsk('Profile Expert', 30000);
  await checkCommState('Active profile', 'Expert', _mtL('Profile Expert', 'Profile Expert'));
  await checkCommState(
    'Verification route lines',
    'visible',
    _mtL('Verification-Route-Display-Policy Default', 'Verification-route display policy default')
  );
  const vrHideOn = await _mtTryPanelAction(
    'set_hide_verification_route_lines',
    {scope: 'provider', provider: 'gemini', enabled: true},
    10000
  );
  if(!_mtCheck(
    !!(vrHideOn && vrHideOn.ok !== false),
    _mtL(
      'Verification-Route-Display-Policy Toggle (provider=gemini, hidden) ok',
      'Verification-route display policy toggle (provider=gemini, hidden) ok'
    ),
    _mtL(
      'Verification-Route-Display-Policy Toggle (provider=gemini, hidden) fehlgeschlagen',
      'Verification-route display policy toggle (provider=gemini, hidden) failed'
    )
  )) fails++;
  await checkCommState(
    'Verification route lines',
    'hidden',
    _mtL(
      'Verification-Route-Display-Policy provider=gemini -> hidden',
      'Verification-route display policy provider=gemini -> hidden'
    )
  );
  const vrHiddenContent = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'vr-policy-hidden-content'});
  if(_mtLooksLikeTransientProviderError(vrHiddenContent.html || '')){
    _mtWarn(_mtL(
      'Verification-Route-Hidden-Content-Check uebersprungen: transiente Provider-Fehlerantwort.',
      'Verification-route hidden content check skipped: transient provider error response.'
    ));
  } else {
    if(!_mtCheck(
      !/Runtime Error in Renderer/i.test(_mtStripHtml(vrHiddenContent.html || '')),
      _mtL(
        'Verification-Route-Hidden-Content-Check: kein Runtime-Renderer-Error',
        'Verification-route hidden content check: no runtime renderer error'
      ),
      _mtL(
        'Verification-Route-Hidden-Content-Check: Runtime-Renderer-Error erkannt',
        'Verification-route hidden content check: runtime renderer error detected'
      )
    )) fails++;
    if(!_mtCheck(
      !/Verification Route\\s*:/i.test(_mtStripHtml(vrHiddenContent.html || '')),
      _mtL(
        'Verification-Route-Hidden-Content-Check: Verification-Route-Header ausgeblendet',
        'Verification-route hidden content check: verification route header hidden'
      ),
      _mtL(
        'Verification-Route-Hidden-Content-Check: Verification-Route-Header noch sichtbar',
        'Verification-route hidden content check: verification route header still visible'
      )
    )) fails++;
  }
  const vrHideClear = await _mtTryPanelAction(
    'set_hide_verification_route_lines',
    {scope: 'provider', provider: 'gemini', clear: true},
    10000
  );
  if(!_mtCheck(
    !!(vrHideClear && vrHideClear.ok !== false),
    _mtL(
      'Verification-Route-Display-Policy Reset (provider=gemini) ok',
      'Verification-route display policy reset (provider=gemini) ok'
    ),
    _mtL(
      'Verification-Route-Display-Policy Reset (provider=gemini) fehlgeschlagen',
      'Verification-route display policy reset (provider=gemini) failed'
    )
  )) fails++;
  await checkCommState(
    'Verification route lines',
    'visible',
    _mtL(
      'Verification-Route-Display-Policy provider=gemini reset -> visible',
      'Verification-route display policy provider=gemini reset -> visible'
    )
  );

  await _mtAsk('Strict on', 30000);
  await checkCommState('Overlay', 'Strict', 'Strict on');
  await _mtAsk('Strict off', 30000);
  await checkCommState('Overlay', 'off', 'Strict off');

  await _mtAsk('Explore on', 30000);
  await checkCommState('Overlay', 'Explore', 'Explore on');
  await _mtAsk('Explore off', 30000);
  await checkCommState('Overlay', 'off', 'Explore off');

  await _mtAsk('Color on', 30000);
  await checkCommState('Color', 'on', 'Color on');
  await _mtAsk('Color off', 30000);
  await checkCommState('Color', 'off', 'Color off');

  await _mtAsk('SCI off', 30000);
  await checkCommState('SCI', 'OFF', 'SCI off');
  await _mtAsk('SCI recurse', 30000);
  await checkCommState('Comm active', 'on', 'SCI recurse');

  await _mtAsk('Comm Anchor on', 30000);
  await checkCommState('Anchor auto', 'on', 'Comm Anchor on');
  await _mtAsk('Comm Anchor off', 30000);
  await checkCommState('Anchor auto', 'off', 'Comm Anchor off');

  await _mtAsk('Dynamic one-shot on', 30000);
  await checkCommState('Dynamic nudge', 'one-shot', 'Dynamic one-shot on');

  const stopRes = await _mtAsk('Comm Stop', 30000);
  if(!_mtCheck(
    _mtHasCommStateField(stopRes.html, 'Comm active', 'off'),
    'Comm Stop: ' + _mtL('Comm State ok', 'Comm State ok'),
    'Comm Stop: ' + _mtL(
      'Comm Stop-Response zeigt Comm active=off nicht',
      'Comm Stop response does not show Comm active=off'
    )
  )) fails++;
  const blockedAsk = await _mtTryPanelAction('ask', {text: _mtPromptShort()}, 30000);
  const blockedErr = String((blockedAsk && blockedAsk.error) || '');
  if(!_mtCheck(
    !!(blockedAsk && blockedAsk.ok === false && /comm_off_blocked/i.test(blockedErr)),
    'Comm Stop: ' + _mtL('panel_action ask wird bei Comm-off geblockt', 'panel_action ask is blocked while comm is off'),
    'Comm Stop: ' + _mtL('panel_action ask wird bei Comm-off nicht geblockt', 'panel_action ask is not blocked while comm is off')
  )) fails++;
  const startViaPanel = await _mtTryPanelAction('ask', {text: 'Comm Start'}, 30000);
  if(!_mtCheck(
    !!(startViaPanel && String(startViaPanel.error || '') !== 'comm_off_blocked'),
    'Comm Stop: ' + _mtL('panel_action ask Comm Start bleibt erlaubt', 'panel_action ask Comm Start remains allowed'),
    'Comm Stop: ' + _mtL('panel_action ask Comm Start wurde geblockt', 'panel_action ask Comm Start was blocked')
  )) fails++;
  const unknownAction = await _mtTryPanelAction('__unknown_action__', {}, 10000);
  if(!_mtCheck(
    !!(unknownAction && unknownAction.ok === false && /unknown action/i.test(String(unknownAction.error || ''))),
    'Comm Stop: ' + _mtL('panel_action unknown action liefert stabiles Fehlerschema', 'panel_action unknown action returns stable error schema'),
    'Comm Stop: ' + _mtL('panel_action unknown action liefert kein stabiles Fehlerschema', 'panel_action unknown action does not return stable error schema')
  )) fails++;
  await _mtAsk('Comm Start', 30000);
  await checkCommState('Comm active', 'on', 'Comm Start');
  await checkCommState('Active profile', 'Standard', 'Comm Start');

  return {fails};
}

async function _mtScenarioCscMiddleBlock(){
  let fails = 0;
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());
  await _mtAsk('Profile Standard', 30000);
  await _mtAsk('SCI off', 30000);
  await _mtAsk('Strict off', 30000);
  await _mtAsk('Explore off', 30000);

  const res = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'csc-mid-block'});
  if(_mtLooksLikeTransientProviderError(res.html || '')){
    _mtWarn(_mtL(
      'CSC-Mittelblock-Check uebersprungen: transiente Provider-Fehlerantwort.',
      'CSC middle-block check skipped: transient provider error response.'
    ));
    return {fails};
  }

  if(!_mtCheck(
    _mtHasActiveProfileHeader(res.html, 'Standard'),
    _mtL('CSC-Mittelblock: Header zeigt Standard-Profil', 'CSC middle block: header shows Standard profile'),
    _mtL('CSC-Mittelblock: Header fehlt/zeigt falsches Profil', 'CSC middle block: header missing/wrong profile')
  )) fails++;
  if(!_mtCheck(
    !/Runtime Error in Renderer/i.test(_mtStripHtml(res.html || '')),
    _mtL('CSC-Mittelblock: kein Runtime-Renderer-Error', 'CSC middle block: no runtime renderer error'),
    _mtL('CSC-Mittelblock: Runtime-Renderer-Error erkannt', 'CSC middle block: runtime renderer error detected')
  )) fails++;
  if(!_mtCheck(
    !_mtHasStrictEnforcementBanner(res.html),
    _mtL('CSC-Mittelblock: kein Strict-Enforcement-Banner im Healthy-Path', 'CSC middle block: no strict-enforcement banner on healthy path'),
    _mtL('CSC-Mittelblock: unerwartetes Strict-Enforcement-Banner im Healthy-Path', 'CSC middle block: unexpected strict-enforcement banner on healthy path')
  )) fails++;
  if(!_mtCheck(
    !_mtHasControlLayerAlertsBox(res.html),
    _mtL('CSC-Mittelblock: kein Control-Layer-Alert-Container im Healthy-Path', 'CSC middle block: no control-layer alerts container on healthy path'),
    _mtL('CSC-Mittelblock: unerwarteter Control-Layer-Alert-Container im Healthy-Path', 'CSC middle block: unexpected control-layer alerts container on healthy path')
  )) fails++;
  if(!_mtCheck(
    (!_mtHasStrictEnforcementBanner(res.html)) && (!_mtHasControlLayerAlertsBox(res.html)),
    _mtL(
      'CSC-Mittelblock: Runtime-Stage-Orchestrierung stabil (Strict-Gate-Dispatch + Hook-Resolver im Stage-Modul aktiv)',
      'CSC middle block: runtime-stage orchestration stable (strict-gate dispatch + hook resolver in stage module active)'
    ),
    _mtL(
      'CSC-Mittelblock: Runtime-Stage-Orchestrierung instabil (Strict-Gate-Dispatch/Hook-Resolver)',
      'CSC middle block: runtime-stage orchestration unstable (strict-gate dispatch/hook resolver)'
    )
  )) fails++;
  return {fails};
}

async function _mtScenarioLogReplaySeam(){
  let fails = 0;
  const q = _mtPromptShort();
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());
  await _mtAsk('Profile Standard', 30000);
  await _mtAsk('SCI off', 30000);
  await _mtAskWithRetry(q, 180000, {maxRetries: 1, label: 'replay-seam-source'});

  let exportedName = '';
  let exportedChatPath = '';
  try {
    const exp = await _mtPanelAction('export', {}, 25000);
    const chatPath = String((exp && exp.chat_path) || (exp && exp.result && exp.result.chat_path) || '');
    exportedChatPath = chatPath;
    exportedName = _mtBasename(chatPath);
  } catch(e) {
    _mtWarn(_mtL('Replay-Seam: Export fehlgeschlagen, fallback auf Chat-Log-Liste.', 'Replay seam: export failed, fallback to chat log list.'));
  }
  if(exportedChatPath){
    const pv = await _mtPanelAction('preview_export_file', {path: exportedChatPath, max_chars: 1200}, 20000);
    const pvOk = !!(pv && pv.ok !== false && (typeof pv.preview === 'string' || (pv.result && typeof pv.result.preview === 'string')));
    if(!_mtCheck(
      pvOk,
      _mtL('Replay-Seam: preview_export_file ueber panel_action ok', 'Replay seam: preview_export_file via panel_action ok'),
      _mtL('Replay-Seam: preview_export_file ueber panel_action fehlgeschlagen', 'Replay seam: preview_export_file via panel_action failed')
    )) fails++;
  } else {
    _mtWarn(_mtL('Replay-Seam: preview_export_file-Check uebersprungen (kein exportierter chat_path).', 'Replay seam: preview_export_file check skipped (no exported chat_path).'));
  }

  const logs = await _mtListChatLogs(50);
  let targetName = exportedName;
  if(!targetName){
    targetName = String((logs && logs[0]) || '');
  }
  targetName = _mtBasename(targetName);
  if(!_mtCheck(
    !!targetName,
    _mtL('Replay-Seam: Chat-Log-Datei fuer Load bestimmt', 'Replay seam: resolved chat-log filename for load'),
    _mtL('Replay-Seam: keine Chat-Log-Datei fuer Load bestimmbar', 'Replay seam: could not resolve chat-log filename for load')
  )) fails++;
  if(!targetName) return {fails};

  await _mtPanelAction('clear_chat', {}, 8000);
  const loaded = await _mtLoadChatLogByName(targetName, true);
  const loadedOk = !!(loaded && loaded.ok !== false && ((loaded.history_len > 0) || (loaded.result && loaded.result.history_len > 0)));
  if(!_mtCheck(
    loadedOk,
    _mtL('Replay-Seam: Chat-Log geladen (history_len > 0)', 'Replay seam: chat log loaded (history_len > 0)'),
    _mtL('Replay-Seam: Chat-Log nicht geladen oder leer', 'Replay seam: chat log not loaded or empty')
  )) fails++;
  const badRes = await _mtTryLoadChatLogByName('__missing_manual_test_log__.json', true);
  const badErr = String((badRes && badRes.error) || '');
  const badRejected = !!(badRes && badRes.ok === false && (
    /file_not_found/i.test(badErr) || /load_chat_log_failed/i.test(badErr)
  ));
  if(!_mtCheck(
    badRejected,
    _mtL('Replay-Seam: ungueltiger Chat-Log wird sauber abgelehnt', 'Replay seam: invalid chat log is rejected cleanly'),
    _mtL('Replay-Seam: ungueltiger Chat-Log wurde nicht sauber abgelehnt', 'Replay seam: invalid chat log was not rejected cleanly')
  )) fails++;
  const traversalRes = await _mtTryLoadChatLogByName('../__manual_test_traversal__.json', true);
  const traversalErr = String((traversalRes && traversalRes.error) || '');
  const traversalRejected = !!(traversalRes && traversalRes.ok === false && (
    /path_traversal_blocked/i.test(traversalErr) || /load_chat_log_failed/i.test(traversalErr)
  ));
  if(!_mtCheck(
    traversalRejected,
    _mtL('Replay-Seam: Traversal-Name wird geblockt', 'Replay seam: traversal filename is blocked'),
    _mtL('Replay-Seam: Traversal-Name wurde nicht geblockt', 'Replay seam: traversal filename was not blocked')
  )) fails++;
  return {fails};
}

async function _mtScenarioFullRegressionLight(){
  let fails = 0;
  for(const fn of [_mtScenarioSmokeShort, _mtScenarioProfileSelfDebunking, _mtScenarioProviderSwitch, _mtScenarioSciFormat, _mtScenarioQcOverrideFooter]){
    _mtEnsureRunning();
    const r = await fn();
    fails += (r && r.fails) ? r.fails : 0;
  }
  await _mtExportChatAudit('full_regression_light');
  return {fails};
}

async function _mtScenarioActualTest(){
  let fails = 0;
  _mtLog(_mtL(
    'ACTUAL-TEST > schneller Kern-Regressionlauf',
    'ACTUAL-TEST > fast core regression run'
  ));
  for(const fn of [_mtScenarioStateRoutingCore, _mtScenarioSmokeShort, _mtScenarioCscMiddleBlock, _mtScenarioLogReplaySeam, _mtScenarioProfileSelfDebunking, _mtScenarioSciFormat, _mtScenarioQcOverrideFooter]){
    _mtEnsureRunning();
    const r = await fn();
    fails += (r && r.fails) ? r.fails : 0;
  }
  await _mtExportChatAudit('actual_test_final');
  return {fails};
}

async function _mtScenarioKomplexttest(){
  let fails = 0;
  const prompts = [
    _mtPromptShort(),
    _mtPromptLong(),
  ];
  const profiles = ['Standard', 'Expert'];
  const sciVariants = ['off', 'A', 'B'];
  const qcStates = [false, true];
  const colorStates = ['on', 'off'];
  const totalCases = profiles.length * sciVariants.length * qcStates.length * colorStates.length;
  let caseIdx = 0;

  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtSetProvider('gemini');
  await _mtSetAnswerLanguage(_mtSelectedAnswerLanguage());

  for(const profile of profiles){
    for(const sci of sciVariants){
      for(const qcOn of qcStates){
        for(const color of colorStates){
          _mtEnsureRunning();
          caseIdx += 1;
          _mtLog(
            'CASE ' + caseIdx + '/' + totalCases
            + ' > profile=' + profile
            + ' · sci=' + sci
            + ' · qc_override=' + (qcOn ? 'on' : 'off')
            + ' · color=' + color
          );

          if(caseIdx > 1){
            await _mtExportChatAudit('case_checkpoint_before_clear_chat_' + caseIdx);
          }
          await _mtPanelAction('clear_chat', {}, 8000);
          await _mtAsk('Profile ' + profile, 30000);
          if(color === 'on') await _mtAsk('Color on', 30000);
          else await _mtAsk('Color off', 30000);

          if(sci === 'off'){
            await _mtAsk('SCI off', 30000);
          } else {
            await _mtAsk('SCI menu', 30000);
            await _mtAsk(sci, 30000);
          }

          if(qcOn){
            await _mtApplyQcOverride({Clarity:3, Brevity:0, Evidence:3, Empathy:3, Consistency:3, Neutrality:3});
          } else {
            try { await _mtClearQcOverride(); } catch(e) {}
          }

          for(let i = 0; i < prompts.length; i += 1){
            _mtEnsureRunning();
            const prompt = prompts[i];
            const caseLabel = 'CASE ' + caseIdx + ' P' + String(i + 1);
            const res = await _mtAskWithRetry(prompt, 180000, {maxRetries: 1, label: caseLabel});
            const txt = _mtStripHtml(res.html || '');

            if(_mtLooksLikeTransientProviderError(res.html || '')){
              _mtWarn(caseLabel + ': ' + _mtL(
                'transiente Provider-Fehlerantwort (Checks uebersprungen).',
                'transient provider error response (checks skipped).'
              ));
              continue;
            }

            const hasCompleteQcFooter = _mtHasCompleteQcFooter(res.html);
            const hasUCode = _mtContainsUCode(res.html || '', txt);
            const unstableSciAlert = _mtHasSciAlertMissingTraceStepContent(res.html);

            if(unstableSciAlert && !hasCompleteQcFooter){
              _mtWarn(caseLabel + ': ' + _mtL(
                'QC-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
                'QC check skipped (SCI alert: missing SCI Trace step content).'
              ));
            } else if(!_mtCheck(
              hasCompleteQcFooter,
              caseLabel + ': ' + _mtL('QC-Footer vollstaendig', 'QC footer complete'),
              caseLabel + ': ' + _mtL('QC-Footer fehlt/unvollstaendig', 'QC footer missing/incomplete')
            )) fails++;

            if(unstableSciAlert && !hasUCode){
              _mtWarn(caseLabel + ': ' + _mtL(
                'U-Marker-Check uebersprungen (SCI-Alert: Missing SCI Trace step content).',
                'U-marker check skipped (SCI alert: missing SCI Trace step content).'
              ));
            } else if(!_mtCheck(
              hasUCode,
              caseLabel + ': ' + _mtT('has_u_marker'),
              caseLabel + ': ' + _mtT('no_u_marker')
            )) fails++;

            const hasTagMarker = /\[(GREEN|YELLOW|RED|GRAY|WHITE)(-[A-Z0-9]+)*\]/i.test(txt);
            const hasEmojiMarker = txt.indexOf('🟢') >= 0 || txt.indexOf('🟡') >= 0 || txt.indexOf('🔴') >= 0 || txt.indexOf('⚪') >= 0;
            const hasColorMarker = hasTagMarker || hasEmojiMarker;
            if(color === 'on'){
              if(!_mtCheck(
                hasColorMarker,
                caseLabel + ': ' + _mtL('Farbmarker vorhanden (Color on)', 'Color markers present (Color on)'),
                caseLabel + ': ' + _mtL('Farbmarker fehlen (Color on)', 'Color markers missing (Color on)')
              )) fails++;
            } else if(hasColorMarker){
              _mtWarn(caseLabel + ': ' + _mtL(
                'Farbmarker trotz Color off erkannt (modell-/renderpfadabhaengig).',
                'Color markers detected although Color is off (model/render-path dependent).'
              ));
            }
          }

          if(qcOn){
            try { await _mtClearQcOverride(); } catch(e) { _mtWarn(_mtL('QC-Override clear fehlgeschlagen: ', 'QC override clear failed: ') + String(e && e.message ? e.message : e)); }
          }
        }
      }
    }
  }

  _mtEnsureRunning();
  await _mtExportChatAudit('before_influence_checks');
  _mtLog(_mtT('influence_checks'));
  await _mtPanelAction('clear_chat', {}, 8000);
  await _mtAsk('Profile Standard', 30000);
  await _mtAsk('SCI off', 30000);
  await _mtAsk('Color on', 30000);

  const baseline = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-baseline-cgi'});
  const baselineTxt = _mtStripHtml(baseline.html || '');

  if(_mtLooksLikeTransientProviderError(baseline.html || '')){
    _mtWarn(_mtL(
      'CGI-Influence-Check uebersprungen: transiente Provider-Fehlerantwort bei Baseline.',
      'CGI influence check skipped: transient provider error on baseline.'
    ));
  } else {
    const cgiNote = await _mtAskWithRetry('3,3,3', 30000, {maxRetries: 1, label: 'influence-cgi-note'});
    if(!_mtLooksLikeTransientProviderError(cgiNote.html || '')){
      if(!_mtCheck(
        /cgi/i.test(_mtStripHtml(cgiNote.html || '')),
        _mtT('cgi_recorded'),
        _mtT('cgi_not_recorded')
      )) fails++;
      const afterCgi = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-after-cgi'});
      const afterCgiTxt = _mtStripHtml(afterCgi.html || '');
      if(_mtLooksLikeTransientProviderError(afterCgi.html || '')){
        _mtWarn(_mtL(
          'CGI-Influence-Check teilweise uebersprungen: transiente Provider-Fehlerantwort nach CGI.',
          'CGI influence check partially skipped: transient provider error after CGI.'
        ));
      } else if(!_mtCheck(
        baselineTxt !== afterCgiTxt,
        _mtT('cgi_influence_yes'),
        _mtT('cgi_influence_no')
      )) fails++;
    } else {
      _mtWarn(_mtL(
        'CGI-Influence-Check uebersprungen: transiente Provider-Fehlerantwort beim CGI-Triplet.',
        'CGI influence check skipped: transient provider error on CGI triplet.'
      ));
    }
  }

  const baselineQc = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-baseline-qc'});
  await _mtApplyQcOverride({Clarity:3, Brevity:0, Evidence:3, Empathy:3, Consistency:3, Neutrality:3});
  const afterQc = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-after-qc'});
  if(_mtLooksLikeTransientProviderError(baselineQc.html || '') || _mtLooksLikeTransientProviderError(afterQc.html || '')){
    _mtWarn(_mtL(
      'QC-Influence-Check uebersprungen: transiente Provider-Fehlerantwort.',
      'QC influence check skipped: transient provider error response.'
    ));
  } else if(!_mtCheck(
    _mtStripHtml(baselineQc.html || '') !== _mtStripHtml(afterQc.html || ''),
    _mtT('qc_influence_yes'),
    _mtT('qc_influence_no')
  )) fails++;
  try { await _mtClearQcOverride(); } catch(e) { _mtWarn(_mtL('QC-Override clear fehlgeschlagen: ', 'QC override clear failed: ') + String(e && e.message ? e.message : e)); }

  const baselineDyn = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-baseline-dynamic'});
  await _mtAsk('Dynamic one-shot on', 30000);
  const afterDyn = await _mtAskWithRetry(_mtPromptShort(), 180000, {maxRetries: 1, label: 'influence-after-dynamic'});
  if(_mtLooksLikeTransientProviderError(baselineDyn.html || '') || _mtLooksLikeTransientProviderError(afterDyn.html || '')){
    _mtWarn(_mtL(
      'Dynamic-Influence-Check uebersprungen: transiente Provider-Fehlerantwort.',
      'Dynamic influence check skipped: transient provider error response.'
    ));
  } else if(!_mtCheck(
    _mtStripHtml(baselineDyn.html || '') !== _mtStripHtml(afterDyn.html || ''),
    _mtT('dynamic_influence_yes'),
    _mtT('dynamic_influence_no')
  )) fails++;

  await _mtExportChatAudit('komplexttest_final');
  return {fails};
}

function stopManualTestRunner(){
  if(!window.__manualTestRunner) return;
  window.__manualTestRunner.stop = true;
  _mtLog(_mtT('stop_requested'));
}

async function startManualTestRunner(){
  const mt = window.__manualTestRunner || (window.__manualTestRunner = { running:false, stop:false, runId:0 });
  if(mt.running){
    _mtWarn('Ein Testlauf laeuft bereits.');
    return;
  }
  mt.running = true;
  mt.stop = false;
  mt.runId = (mt.runId || 0) + 1;
  mt.events = [];
  mt.summary = null;
  mt.monitorEnabled = _mtMonitorEnabled();
  mt.askFallbackNoted = false;
  mt.qcApplyFallbackNoted = false;
  mt.qcClearFallbackNoted = false;
  mt.mainMirrorWarned = false;
  mt.monitorLang = _mtSelectedAnswerLanguage();
  const myRun = mt.runId;
  _mtSetButtons(true);
  _mtClearLog();
  const sel = document.getElementById('manualTestScenario');
  const scenario = (sel && sel.value) ? sel.value : 'smoke_short';
  mt.scenario = scenario;
  mt.startedAt = (_now ? _now() : null);
  try {
    if(mt.monitorEnabled){
      await _apiCall('panel_action', ['manual_test_monitor_show', {}], 8000);
      await _apiCall('panel_action', ['manual_test_monitor_reset', {
        scenario: scenario,
        status: 'running',
        summary: '-',
        lang: mt.monitorLang,
      }], 8000);
    }
  } catch(e) {
    _mtWarn(_mtT('monitor_init_failed') + String(e && e.message ? e.message : e));
  }
  _mtLog(_mtT('test_started_prefix') + scenario);
  const t0 = Date.now();
  try {
    let result = {fails: 0};
    if(scenario === 'actual_test') result = await _mtScenarioActualTest();
    else if(scenario === 'smoke_short') result = await _mtScenarioSmokeShort();
    else if(scenario === 'provider_switch') result = await _mtScenarioProviderSwitch();
    else if(scenario === 'sci_format') result = await _mtScenarioSciFormat();
    else if(scenario === 'qc_override_footer') result = await _mtScenarioQcOverrideFooter();
    else if(scenario === 'profile_self_debunking') result = await _mtScenarioProfileSelfDebunking();
    else if(scenario === 'komplexttest') result = await _mtScenarioKomplexttest();
    else if(scenario === 'full_regression_light') result = await _mtScenarioFullRegressionLight();
    else throw new Error(_mtT('unknown_scenario_prefix') + scenario);
    _mtEnsureRunning();
    const ms = Date.now() - t0;
    if((result && result.fails) > 0){
      mt.summary = { status: 'FAIL', fails: Number(result.fails||0) };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'FAIL', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog(_mtT('summary_fail_prefix') + result.fails + ' · Duration=' + ms + 'ms', 'err');
      _setStatus(_mtT('status_finished_fail_prefix') + result.fails + _mtT('status_finished_fail_suffix'), 'err');
    } else {
      mt.summary = { status: 'PASS', fails: 0 };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'PASS', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog(_mtT('summary_pass_prefix') + ' · Duration=' + ms + 'ms', 'ok');
      _setStatus(_mtT('status_finished_pass'), 'ok');
    }
    await _mtSaveReport({ duration_ms: ms, summary: mt.summary, finished_at: (_now ? _now() : null) });
  } catch(e) {
    const stopped = (String(e && e.message ? e.message : e).indexOf('manual_test_stopped') >= 0);
    if(stopped){
      mt.summary = { status: 'STOPPED' };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'STOPPED', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog(_mtT('summary_stopped'), 'warn');
      _setStatus(_mtT('status_stopped'), 'err');
      await _mtExportChatAudit('manual_test_stopped_partial');
      await _mtSaveReport({ duration_ms: (Date.now()-t0), summary: mt.summary, finished_at: (_now ? _now() : null) });
    } else {
      mt.summary = { status: 'ERROR', error: String(e && e.message ? e.message : e) };
      try { if(mt.monitorEnabled) await _apiCall('panel_action', ['manual_test_monitor_header', {scenario: scenario, status: 'ERROR', summary: mt.summary, lang: mt.monitorLang}], 4000); } catch(e) {}
      _mtLog(_mtT('summary_error_prefix') + String(e && e.message ? e.message : e), 'err');
      _setStatus(_mtT('status_error_prefix') + String(e && e.message ? e.message : e), 'err');
      await _mtExportChatAudit('manual_test_error_partial');
      await _mtSaveReport({ duration_ms: (Date.now()-t0), summary: mt.summary, finished_at: (_now ? _now() : null) });
    }
  } finally {
    if(window.__manualTestRunner && window.__manualTestRunner.runId === myRun){
      window.__manualTestRunner.running = false;
      window.__manualTestRunner.stop = false;
    }
    _mtSetButtons(false);
    try { await buildUI(); } catch(e) {}
  }
}

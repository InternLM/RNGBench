"use strict";
const $ = (id) => document.getElementById(id);
const state = { games: [], current: null, session: null, done: false, auto: false, busy: false };

const el = {
  switch: $("gameswitch"), desc: $("game-desc"), config: $("game-config"),
  board: $("board"), empty: $("screen-empty"), phase: $("phase"), info: $("info-line"),
  chips: $("chips"), log: $("log"), manual: $("manual"),
  neu: $("btn-new"), step: $("btn-step"), auto: $("btn-auto"), clear: $("btn-clear"),
  base: $("m-base"), key: $("m-key"), model: $("m-model"), temp: $("m-temp"), maxtok: $("m-maxtok"),
};

// ── model config persistence (local only) ─────────────────────────────────────
const SAVE = ["base", "model", "temp", "maxtok"];
function loadCfg(){ try{ const c=JSON.parse(localStorage.getItem("rngbench_cfg")||"{}");
  SAVE.forEach(k=>{ if(c[k]!=null) el[k].value=c[k]; }); }catch(e){} }
function saveCfg(){ const c={}; SAVE.forEach(k=>c[k]=el[k].value);
  localStorage.setItem("rngbench_cfg", JSON.stringify(c)); }

async function api(path, body){
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const d = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}
const esc = s => (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// ── game switcher + dynamic config form ───────────────────────────────────────
async function init(){
  state.games = await (await fetch("/api/games")).json();
  el.switch.innerHTML = "";
  state.games.forEach((g,i)=>{
    const b=document.createElement("button");
    b.className="gtab"+(i===0?" active":""); b.textContent=g.name;
    b.onclick=()=>selectGame(g.id);
    el.switch.appendChild(b);
  });
  if(state.games.length) selectGame(state.games[0].id);
  loadCfg(); logEmpty();
}
function selectGame(id){
  state.current = state.games.find(g=>g.id===id);
  [...el.switch.children].forEach(b=> b.classList.toggle("active", b.textContent===state.current.name));
  el.desc.textContent = state.current.description;
  // build config form
  el.config.innerHTML="";
  state.current.config.forEach(f=>{
    const wrap=document.createElement("label");
    if(f.type==="bool"){
      wrap.className="check";
      wrap.innerHTML=`<input type="checkbox" id="cfg-${f.key}" ${f.default?"checked":""}> ${f.label}`;
    } else if(f.type==="select"){
      wrap.innerHTML=`${f.label}<select id="cfg-${f.key}">`+
        f.options.map(o=>`<option ${o===String(f.default)?"selected":""}>${o}</option>`).join("")+`</select>`;
    } else {
      wrap.innerHTML=`${f.label}<input type="number" id="cfg-${f.key}" value="${f.default??0}">`;
    }
    el.config.appendChild(wrap);
  });
  // reset session state on game switch
  state.session=null; state.done=false; stopAuto();
  el.step.disabled=true; el.auto.disabled=true; el.manual.innerHTML="";
  el.board.removeAttribute("src"); el.empty.style.display=""; setPhase("idle");
  el.info.textContent=""; el.chips.innerHTML="";
}
function readConfig(){
  const cfg={};
  state.current.config.forEach(f=>{
    const node=$("cfg-"+f.key); if(!node) return;
    cfg[f.key] = f.type==="bool" ? node.checked : (f.type==="int" ? parseInt(node.value) : node.value);
  });
  return cfg;
}

// ── rendering ─────────────────────────────────────────────────────────────────
function setPhase(p){
  const map={idle:"idle", running:"running", done:"game over",
    flip_first:"flip 1", flip_second:"flip 2"};
  el.phase.className = "screen-tag "+(state.done?"done":(state.auto?"running":(p||"")));
  el.phase.textContent = state.done ? "game over" : (map[p]||p||"ready");
}
function setBoard(view){ if(view&&view.board){ el.board.src=view.board; el.empty.style.display="none"; } }
function renderChips(stats){
  if(!stats) return;
  el.chips.innerHTML = Object.entries(stats).map(([k,v])=>
    `<div class="chip"><span class="chip-num">${esc(String(v))}</span><span class="chip-lab">${esc(k)}</span></div>`).join("");
}
function renderManual(actions){
  el.manual.innerHTML="";
  (actions||[]).forEach(a=>{
    const b=document.createElement("button"); b.className="mbtn"; b.textContent=a.label;
    b.onclick=()=>manual(a.action); el.manual.appendChild(b);
  });
}
function logEmpty(){ el.log.innerHTML='<div class="log-empty">No calls yet. Press <b>Auto-play</b> or <b>Step</b> to let the model play.</div>'; }
function addEntry(html, cls=""){
  if(el.log.querySelector(".log-empty")) el.log.innerHTML="";
  const d=document.createElement("div"); d.className="entry "+cls; d.innerHTML=html; el.log.prepend(d);
}
function logResult(out){
  (out.log||[]).forEach(e=>{
    const act = e.coord || e.action || "—";
    const res = e.result||"";
    const retry = e.attempt>0 ? `<span class="tag retry">retry ${e.attempt}</span>` : "";
    const info = e.info ? ` <span class="hint">${esc(e.info)}</span>` : "";
    addEntry(`<div class="meta"><span class="tag ${res}">${esc(res)}</span>`+
      `<span class="code">${esc(act)}</span>${retry}${info}</div>`+
      (e.raw?`<div class="raw">${esc(e.raw)}</div>`:""));
  });
  if(out.round_done && out.verdict){
    const v=out.verdict;
    addEntry(`<div class="round-sep"><span class="tag ${v}">${esc(v.replace("_"," "))}</span></div>`);
  }
}
function applyOut(out){
  setBoard(out.view); renderChips(out.stats);
  state.done = !!out.done;
  setPhase(out.phase);
  if(state.done){ stopAuto(); el.step.disabled=true; el.auto.disabled=true; }
}

// ── model config ──────────────────────────────────────────────────────────────
function modelReq(){ return { session:state.session, api_base:el.base.value.trim(),
  api_key:el.key.value, model:el.model.value.trim(),
  temperature:parseFloat(el.temp.value)||0.7, max_tokens:parseInt(el.maxtok.value)||2048 }; }
function validModel(){
  if(!el.base.value.trim()){ alert("Enter the server URL."); return false; }
  if(!el.model.value.trim()){ alert("Enter the model name."); return false; }
  return true;
}

// ── actions ───────────────────────────────────────────────────────────────────
async function newGame(){
  setBusy(true);
  try{
    const out = await api("/api/new", { game:state.current.id, config:readConfig() });
    state.session=out.session; state.done=false;
    setBoard(out.view); renderChips(out.stats); renderManual(out.actions);
    el.info.textContent=out.info||""; setPhase(out.phase);
    el.step.disabled=false; el.auto.disabled=false; logEmpty();
    addEntry(`<span class="hint">new ${esc(out.name)} · ${esc(out.info)}</span>`,"");
  }catch(e){ alert("Failed to start: "+e.message); }
  finally{ setBusy(false); }
}
async function oneStep(){
  if(!state.session||state.done||!validModel()) return;
  saveCfg(); setBusy(true);
  try{ const out=await api("/api/step", modelReq()); logResult(out); applyOut(out); }
  catch(e){ addEntry("Error: "+esc(e.message),"err"); stopAuto(); }
  finally{ setBusy(false); }
}
async function autoLoop(){
  while(state.auto && !state.done){
    if(!validModel()){ stopAuto(); return; }
    saveCfg();
    try{ const out=await api("/api/step", modelReq()); logResult(out); applyOut(out); }
    catch(e){ addEntry("Error: "+esc(e.message),"err"); stopAuto(); return; }
    await new Promise(r=>setTimeout(r,350));
  }
  stopAuto();
}
function startAuto(){ if(!state.session||state.done||!validModel()) return;
  state.auto=true; el.auto.textContent="Stop"; el.auto.classList.add("running");
  el.step.disabled=true; setPhase("running"); autoLoop(); }
function stopAuto(){ state.auto=false; el.auto.textContent="Auto-play"; el.auto.classList.remove("running");
  if(state.session&&!state.done) el.step.disabled=false; }
async function manual(action){
  if(!state.session||state.done) return;
  setBusy(true);
  try{ const out=await api("/api/manual", {session:state.session, action}); logResult(out); applyOut(out); }
  catch(e){ addEntry("Error: "+esc(e.message),"err"); }
  finally{ setBusy(false); }
}
function setBusy(b){ state.busy=b; el.neu.disabled=b;
  if(!state.auto) el.step.disabled = b||!state.session||state.done;
  [...el.manual.children].forEach(x=>x.disabled=b||state.done); }

// ── wire up ───────────────────────────────────────────────────────────────────
el.neu.onclick=newGame; el.step.onclick=oneStep;
el.auto.onclick=()=> state.auto?stopAuto():startAuto();
el.clear.onclick=logEmpty;
init();

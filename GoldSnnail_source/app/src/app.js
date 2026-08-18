import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

async function invokeCmd(name, args) {
  return await window.__TAURI__.invoke(name, args);
}

const mockArcInput = [
  0,0,0,0,0,0,0,0,0,0,
  0,3,0,0,0,0,2,2,0,0,
  0,3,3,0,0,0,2,2,2,0,
  0,0,3,0,0,0,0,2,0,0,
  0,0,0,0,0,0,0,0,0,0,
  0,0,1,1,0,0,0,0,0,0,
  0,0,1,0,0,0,0,0,0,0,
  0,0,0,0,0,4,4,4,4,0,
  0,0,0,0,0,4,4,4,0,0,
  0,0,0,0,0,0,4,0,0,0
];

function renderArcGrid(id, data) {
  const container = document.getElementById(id);
  container.innerHTML = '';
  data.forEach((val, idx) => {
    const cell = document.createElement('div');
    cell.className = `arc-cell c${val}`;
    cell.onclick = () => {
      if (window.injectSpecificBurst) window.injectSpecificBurst(idx);
      cell.style.transform = 'scale(0.8)';
      setTimeout(() => cell.style.transform = 'scale(1)', 150);
    };
    container.appendChild(cell);
  });
}

async function executeDsl(op) {
  if (window.injectBurst) window.injectBurst();
  try {
    const output = await invokeCmd('solve_arc_task', {
      inputGrid: mockArcInput,
      operation: op
    });
    renderArcGrid('arc-output', output);
  } catch (e) {
    console.error('ARC solve failed:', e);
  }
}

window.executeDsl = executeDsl;

window.switchTab = function(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'monster' && !window.monsterInitialized) initMonster();
};

let snnState = null;
let running = false, tick = 0, animId;
let history = { spikes:[], rate:[], synapses:[], weight:[] };
let cam = { x:0, y:0, zoom:1, dragging:false, lastX:0, lastY:0 };
let hovered = null;

const canvas = document.getElementById('snnCanvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const logEl = document.getElementById('log');
const MAX_HISTORY = 60;

function dprSize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function buildPipeline() {
  const STAGES = [
    { key:'sensor', name:'Sensor', color:'#00d4ff' },
    { key:'attention', name:'Attention', color:'#ff4d6d' },
    { key:'memory', name:'Memory', color:'#2ecc71' },
    { key:'compression', name:'Compression', color:'#a67cff' },
    { key:'world', name:'World model', color:'#f1c40f' },
    { key:'rl', name:'RL agent', color:'#e67e22' }
  ];
  const el = document.getElementById('pipeline');
  el.innerHTML = STAGES.map((s,i) => `
    <div class="stage-pill" id="pill-${s.key}" data-idx="${i}">
      <span class="st-name">${s.name}</span>
      <span class="st-count">30</span>
    </div>
    ${i < STAGES.length-1 ? '<span class="arrow-tiny">→</span>' : ''}
  `).join('');
}

async function initNetwork() {
  const rect = dprSize();
  const density = parseFloat(document.getElementById('density').value);
  try {
    snnState = await invokeCmd('init_snn_core', {
      density: density
    });
  } catch (e) {
    console.error('init_snn failed:', e);
    return;
  }
  tick = 0;
  history = { spikes:[], rate:[], synapses:[], weight:[] };
  log('Initialized: ' + snnState.neurons.length + ' neurons, ' + snnState.synapses.length + ' synapses');
  updateMetricsFromResult({
    spike_count: 0,
    active_synapses: snnState.synapses.filter(s => s.weight > 0.05).length,
    mean_weight: snnState.synapses.length > 0 ? snnState.synapses.reduce((a,s) => a + s.weight, 0) / snnState.synapses.length : 0,
    log_lines: ['Initialized']
  });
  draw();
}

function step() {
  if (!snnState) return;
  const thresh = parseFloat(document.getElementById('thresh').value);
  const leak = parseFloat(document.getElementById('leak').value);
  const noise = parseFloat(document.getElementById('noise').value);
  invokeCmd('step_snn', {
    state: snnState,
    input_spikes: []
  }).then(result => {
    snnState = result.state;
    tick = snnState.tick;
    updateMetricsFromResult(result);
    const activeStage = Math.floor((tick / 20) % 6);
    const stages = ['sensor','attention','memory','compression','world','rl'];
    stages.forEach((s, i) => {
      const pill = document.getElementById('pill-' + s);
      if (pill) pill.classList.toggle('active', i === activeStage);
    });
  }).catch(e => console.error('run_snn_step failed:', e));
}

function updateMetricsFromResult(result) {
  document.getElementById('m-spikes').textContent = result.spike_count;
  document.getElementById('m-rate').textContent = (result.spike_count / Math.max(1, snnState.neurons.length) * 100).toFixed(1);
  document.getElementById('m-synapses').textContent = result.active_synapses;
  document.getElementById('m-plasticity').textContent = result.mean_weight.toFixed(2);

  history.spikes.push(result.spike_count);
  history.rate.push((result.spike_count / Math.max(1, snnState.neurons.length)) * 100);
  history.synapses.push(result.active_synapses);
  history.weight.push(result.mean_weight);
  for (let k in history) if (history[k].length > MAX_HISTORY) history[k].shift();

  drawSpark('spark-spikes', history.spikes, '#00d4ff');
  drawSpark('spark-rate', history.rate, '#ff4d6d');
  drawSpark('spark-syn', history.synapses, '#2ecc71');
  drawSpark('spark-plastic', history.weight, '#a67cff');

  if (result.log_lines && result.log_lines.length > 0) {
    log(result.log_lines[0]);
  }
}

function drawSpark(id, data, color) {
  const svg = document.getElementById(id);
  if (!data.length) return;
  const max = Math.max(...data, 0.001), min = Math.min(...data, 0);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (MAX_HISTORY - 1)) * 100;
    const y = 30 - ((v - min) / range) * 28;
    return `${x},${y}`;
  }).join(' ');
  svg.innerHTML = `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
    <polyline points="0,32 ${pts} 100,32" fill="${color}" opacity="0.06" stroke="none"/>`;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!snnState) return;

  ctx.save();
  ctx.translate(cam.x + rect.width / 2, cam.y + rect.height / 2);
  ctx.scale(cam.zoom, cam.zoom);
  ctx.translate(-rect.width / 2, -rect.height / 2);

  for (const syn of snnState.synapses) {
    if (syn.weight < 0.05) continue;
    const a = snnState.neurons[syn.from];
    const b = snnState.neurons[syn.to];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    const alpha = Math.min(0.3, syn.weight * 0.35);
    ctx.strokeStyle = `rgba(100,100,120,${alpha})`;
    ctx.lineWidth = Math.max(0.4, syn.weight);
    ctx.stroke();
  }

  for (const sp of snnState.pending_spikes) {
    const a = snnState.neurons[sp.from];
    const b = snnState.neurons[sp.to];
    if (!a || !b) continue;
    const sp_delay = sp.delay as number;
    const t = sp_delay > 0 ? (tick as number - sp.launched as number) / sp_delay : 0;
    if (t < 0 || t > 1) continue;
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }

  for (const n of snnState.neurons) {
    const isHover = hovered === n.id;
    const r = isHover ? 6.5 : (n.refractory > 0 ? 3.5 : 4);
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    const stageColors = ['#00d4ff', '#ff4d6d', '#2ecc71', '#a67cff', '#f1c40f', '#e67e22'];
    if (tick - n.last_spike < 4) {
      ctx.fillStyle = stageColors[n.stage] || '#00d4ff';
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = 12;
    } else {
      ctx.fillStyle = isHover ? '#fff' : '#444';
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
    if (isHover) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
      ctx.strokeStyle = '#555';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
  ctx.restore();
}

function loop() {
  if (running) step();
  draw();
  animId = requestAnimationFrame(loop);
}

function toggleSim() {
  running = !running;
  document.getElementById('btn-play').textContent = running ? '⏸ Pause' : '▶ Start';
  log(running ? 'Simulation resumed' : 'Simulation paused');
}

function resetSim() {
  running = false;
  document.getElementById('btn-play').textContent = '▶ Start';
  initNetwork();
}

window.toggleSim = toggleSim;
window.resetSim = resetSim;

window.injectBurst = function() {
  if (!snnState) return;
  for (let i = 0; i < 15; i++) {
    const idx = Math.floor(Math.random() * snnState.neurons.length);
    snnState.neurons[idx].v_m = snnState.neurons[idx].threshold + 0.6;
  }
  log('DSL Operation triggered an SNN burst (15 neurons)');
  if (!running) draw();
}

window.injectSpecificBurst = function(gridIdx) {
  if (!snnState) return;
  const sensoryNeurons = snnState.neurons.filter(n => n.stage === 0);
  if (sensoryNeurons.length > 0) {
    const target = sensoryNeurons[gridIdx % sensoryNeurons.length];
    target.v_m = target.threshold + 0.8;
    log(`ARC Pixel ${gridIdx} injected as spike into Sensor node ${target.id}`);
  }
}

function pruneSynapses() {
  if (!snnState) return;
  const before = snnState.synapses.length;
  snnState.synapses = snnState.synapses.filter(s => s.weight > 0.08);
  log('Pruned ' + (before - snnState.synapses.length) + ' weak synapses');
  updateMetricsFromResult({
    spike_count: 0,
    active_synapses: snnState.synapses.filter(s => s.weight > 0.05).length,
    mean_weight: snnState.synapses.length > 0 ? snnState.synapses.reduce((a, s) => a + s.weight, 0) / snnState.synapses.length : 0,
    log_lines: ['Pruned']
  });
}

window.pruneSynapses = pruneSynapses;

function log(msg) {
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-t">[${t}]</span> <span class="log-s">t${tick}</span> <span>${msg}</span>`;
  logEl.insertBefore(line, logEl.firstChild);
  if (logEl.children.length > 50) logEl.lastChild.remove();
}

canvas.addEventListener('mousemove', e => {
  if (!snnState) return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left - cam.x - rect.width / 2) / cam.zoom + rect.width / 2;
  const my = (e.clientY - rect.top - cam.y - rect.height / 2) / cam.zoom + rect.height / 2;
  hovered = null;
  for (const n of snnState.neurons) {
    const dx = n.x - mx, dy = n.y - my;
    if (dx * dx + dy * dy < 100) {
      hovered = n.id;
      tooltip.style.opacity = '1';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 30) + 'px';
      tooltip.innerHTML = `<b>Neuron ${n.id}</b><br>Stage: ${n.stage}<br>V: ${n.v_m.toFixed(3)}<br>Thresh: ${n.threshold.toFixed(2)}`;
      break;
    }
  }
  if (!hovered) tooltip.style.opacity = '0';
});

canvas.addEventListener('mousedown', e => {
  cam.dragging = true;
  cam.lastX = e.clientX;
  cam.lastY = e.clientY;
});
window.addEventListener('mouseup', () => cam.dragging = false);
canvas.addEventListener('mousemove', e => {
  if (!cam.dragging) return;
  cam.x += e.clientX - cam.lastX;
  cam.y += e.clientY - cam.lastY;
  cam.lastX = e.clientX;
  cam.lastY = e.clientY;
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  cam.zoom *= e.deltaY < 0 ? 1.05 : 0.95;
  cam.zoom = Math.max(0.3, Math.min(3.0, cam.zoom));
}, { passive: false });

window.addEventListener('resize', () => { dprSize(); draw(); });

buildPipeline();
dprSize();
initNetwork();
loop();

// Monster Group 3D
let monsterInitialized = false;
let monsterScene, monsterCamera, monsterRenderer, monsterControls;
let monsterLines = null, monsterAutoRotate = true, monsterShowConnections = false;
let monsterPointsData = [];

function initMonster() {
  window.monsterInitialized = true;
  const container = document.getElementById('monster-container');
  monsterScene = new THREE.Scene();
  monsterScene.background = new THREE.Color(0x0a0a0a);

  monsterCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 100);
  monsterCamera.position.set(3, 3, 4);

  monsterRenderer = new THREE.WebGLRenderer({ antialias: true });
  monsterRenderer.setSize(container.clientWidth, container.clientHeight);
  monsterRenderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(monsterRenderer.domElement);

  monsterControls = new OrbitControls(monsterCamera, monsterRenderer.domElement);
  monsterControls.enableDamping = true;
  monsterControls.dampingFactor = 0.05;

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  monsterScene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 5, 5);
  monsterScene.add(dirLight);

  invokeCmd('get_monster_points').then(points => {
    monsterPointsData = points;
    buildMonsterScene(points);
  }).catch(e => console.error('get_monster_points failed:', e));
}

function buildMonsterScene(points) {
  if (!monsterScene) return;

  const radii = points.map(p => Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
  const maxR = Math.max(...radii, 0.001);

  const sphereGroup = new THREE.Group();
  points.forEach((p, i) => {
    const t = radii[i] / maxR;
    const sphereGeo = new THREE.SphereGeometry(0.04 + t * 0.03, 12, 12);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(0.6 - t * 0.6, 0.9, 0.5 + t * 0.3),
      roughness: 0.3,
      metalness: 0.7
    });
    const sphere = new THREE.Mesh(sphereGeo, sphereMat);
    sphere.position.set(p[0], p[1], p[2]);
    sphereGroup.add(sphere);
  });
  monsterScene.add(sphereGroup);

  const axesHelper = new THREE.AxesHelper(2.5);
  monsterScene.add(axesHelper);
  const gridHelper = new THREE.GridHelper(5, 20, 0x333333, 0x1a1a1a);
  gridHelper.position.y = -2.2;
  monsterScene.add(gridHelper);

  function animateMonster() {
    requestAnimationFrame(animateMonster);
    if (monsterAutoRotate) monsterScene.rotation.y += 0.003;
    monsterControls.update();
    monsterRenderer.render(monsterScene, monsterCamera);
  }
  animateMonster();
}

window.monsterResetView = () => {
  monsterCamera.position.set(3, 3, 4);
  monsterControls.target.set(0, 0, 0);
  if (monsterScene) monsterScene.rotation.y = 0;
};

window.monsterToggleRotate = () => {
  monsterAutoRotate = !monsterAutoRotate;
};

window.monsterToggleConnections = () => {
  monsterShowConnections = !monsterShowConnections;
  if (monsterShowConnections) window.monsterRebuildConnections();
  else if (monsterLines) {
    monsterScene.remove(monsterLines);
    monsterLines = null;
  }
};

window.monsterRebuildConnections = () => {
  if (!monsterShowConnections || !monsterScene) return;
  if (monsterLines) monsterScene.remove(monsterLines);
  if (monsterPointsData.length < 2) return;
  const threshold = parseFloat(document.getElementById('conn-thresh').value);
  const material = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.3 });
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  for (let i = 0; i < monsterPointsData.length; i++) {
    for (let j = i + 1; j < monsterPointsData.length; j++) {
      const dx = monsterPointsData[i][0] - monsterPointsData[j][0];
      const dy = monsterPointsData[i][1] - monsterPointsData[j][1];
      const dz = monsterPointsData[i][2] - monsterPointsData[j][2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < threshold) {
        positions.push(monsterPointsData[i][0], monsterPointsData[i][1], monsterPointsData[i][2]);
        positions.push(monsterPointsData[j][0], monsterPointsData[j][1], monsterPointsData[j][2]);
      }
    }
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  monsterLines = new THREE.LineSegments(geometry, material);
  monsterScene.add(monsterLines);
};

// ============================================================================
// Universal DataType Interface
// ============================================================================

window.dtSupportedTypes = [];
window.dtCurrentTab = 'hex';
window.dtLastEncoded = null;

async function dtInit() {
  try {
    window.dtSupportedTypes = await invokeCmd('list_supported_types');
    const list = document.getElementById('dt-type-list');
    if (list) {
      list.innerHTML = window.dtSupportedTypes.map(t =>
        `<span class="dt-type-tag" title="${t.description}">0x${t.tag.toString(16).padStart(2,'0')} ${t.name}</span>`
      ).join('');
    }
  } catch (e) {
    console.error('Failed to load supported types:', e);
  }
}

window.dtSwitchTab = function(tab) {
  window.dtCurrentTab = tab;
  document.querySelectorAll('.dt-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.dt-panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('dt-panel-' + tab).classList.add('active');
};

window.dtEncode = async function() {
  const type = document.getElementById('dt-encode-type').value;
  let result;
  try {
    switch (type) {
      case 'spike': {
        const events = [
          { src: 0, dst: 10, delay_ticks: 3, amplitude_u8: 200, flags: 0 },
          { src: 5, dst: 20, delay_ticks: 1, amplitude_u8: 100, flags: 1 },
        ];
        result = await invokeCmd('encode_spike_stream', { events });
        break;
      }
      case 'arena': {
        result = await invokeCmd('encode_state_arena', {
          membrane: [0.1, -0.2, 0.3],
          recovery: [0.0, 0.0, 0.0],
          threshold: [-55.0, -55.0, -55.0],
          refractory: [0, 0, 0],
        });
        break;
      }
      case 'weight': {
        result = await invokeCmd('encode_weight_matrix', {
          data: [1.0, 0.5, -0.2, 0.8],
          rows: 2,
          cols: 2,
        });
        break;
      }
      case 'arc': {
        result = await invokeCmd('encode_arc_grid', {
          data: [[0,1,2],[3,4,5],[6,7,8]],
          width: 3,
          height: 3,
        });
        break;
      }
      case 'hyperbolic': {
        result = await invokeCmd('encode_hyperbolic_point', { coords: [0.1, -0.2, 0.3] });
        break;
      }
      case 'quaternion': {
        result = await invokeCmd('encode_quaternion', { w: 1.0, x: 0.5, y: -0.3, z: 0.8 });
        break;
      }
      case 'dvs': {
        result = await invokeCmd('encode_dvs_batch', {
          events: [
            { x: 10, y: 20, polarity: 0, timestamp_us: 1000 },
            { x: 20, y: 30, polarity: 1, timestamp_us: 2000 },
          ]
        });
        break;
      }
      case 'lexicon': {
        result = await invokeCmd('encode_lexicon_token', {
          id: 42,
          surface: 'hello',
          class: 'NounConcrete',
          embedding: { w: 0.9, x: 0.1, y: 0.2, z: 0.3 },
          hyperbolic: { coords: [0.5, 0.5] },
          salience: 0.8,
        });
        break;
      }
    }
    window.dtLastEncoded = result;
    dtRenderResult(result);
    log('Encoded ' + result.type_name + ': ' + result.size_bytes + ' bytes');
  } catch (e) {
    console.error('Encode failed:', e);
    log('Encode failed: ' + e);
  }
};

window.dtDecode = async function() {
  const input = document.getElementById('dt-decode-input').value.trim();
  if (!input) { log('Please paste a hex or base64 payload'); return; }
  try {
    const result = await invokeCmd('decode_payload', { hex: input });
    const json = await invokeCmd('decode_to_json', { hex: input });
    document.getElementById('dt-result-type').textContent = result.type_name;
    document.getElementById('dt-result-size').textContent = input.length + ' chars input';
    document.getElementById('dt-json-output').textContent = json;
    document.getElementById('dt-hex-output').textContent = input;
    document.getElementById('dt-base64-output').textContent = input;
    document.getElementById('dt-raw-output').textContent = json;
    log('Decoded: ' + result.type_name + ' — ' + result.summary);
  } catch (e) {
    console.error('Decode failed:', e);
    log('Decode failed: ' + e);
  }
};

function dtRenderResult(result) {
  document.getElementById('dt-result-type').textContent = result.type_name;
  document.getElementById('dt-result-size').textContent = result.size_bytes + ' bytes';
  document.getElementById('dt-hex-output').textContent = result.hex || '(binary)';
  document.getElementById('dt-base64-output').textContent = result.base64 || '(binary)';
  document.getElementById('dt-json-output').textContent = '(encode a type to see JSON summary)';
  document.getElementById('dt-raw-output').textContent = result.hex || '(binary)';
}

// Initialize on load
setTimeout(dtInit, 100);

// ============================================================================
// ARC Debugger
// ============================================================================

let arcDebugTasks = [];
let arcDebugProgram = [];
let arcDebugStepIndex = 0;
let arcDebugCurrentGrid = null;
let arcDebugPlaying = false;

const ARC_OPS = [
  { code: 0, name: 'Identity', params: [] },
  { code: 1, name: 'Rotate 90°', params: [] },
  { code: 2, name: 'Rotate 180°', params: [] },
  { code: 3, name: 'Rotate 270°', params: [] },
  { code: 4, name: 'Flip H', params: [] },
  { code: 5, name: 'Flip V', params: [] },
  { code: 6, name: 'Move', params: [
    { key: 'dx', label: 'dx', min: -30, max: 30, val: 1 },
    { key: 'dy', label: 'dy', min: -30, max: 30, val: 0 }
  ]},
  { code: 7, name: 'Fill', params: [
    { key: 'color', label: 'color', min: 0, max: 9, val: 5 },
    { key: 'x', label: 'x', min: 0, max: 29, val: 0 },
    { key: 'y', label: 'y', min: 0, max: 29, val: 0 },
    { key: 'w', label: 'w', min: 1, max: 30, val: 2 },
    { key: 'h', label: 'h', min: 1, max: 30, val: 2 }
  ]},
  { code: 8, name: 'Copy', params: [
    { key: 'sx', label: 'sx', min: 0, max: 29, val: 0 },
    { key: 'sy', label: 'sy', min: 0, max: 29, val: 0 },
    { key: 'dx', label: 'dx', min: 0, max: 29, val: 1 },
    { key: 'dy', label: 'dy', min: 0, max: 29, val: 1 },
    { key: 'w', label: 'w', min: 1, max: 30, val: 1 },
    { key: 'h', label: 'h', min: 1, max: 30, val: 1 }
  ]},
  { code: 9, name: 'Gravity Down', params: [] },
  { code: 10, name: 'Gravity Up', params: [] },
  { code: 11, name: 'Gravity Left', params: [] },
  { code: 12, name: 'Gravity Right', params: [] },
  { code: 13, name: 'Mirror', params: [
    { key: 'ax', label: 'ax', min: 0, max: 29, val: 1 },
    { key: 'ay', label: 'ay', min: 0, max: 29, val: 1 }
  ]}
];

window.arcDebugInit = async function() {
  try {
    arcDebugTasks = await invokeCmd('list_arc_tasks');
    const sel = document.getElementById('arc-debug-task');
    sel.innerHTML = arcDebugTasks.map(id => `<option value="${id}">${id}</option>`).join('');
    if (arcDebugTasks.length > 0) arcDebugLoadTask(arcDebugTasks[0]);
  } catch (e) {
    console.error('Failed to load ARC tasks:', e);
  }
};

window.arcDebugLoadTask = async function(taskId) {
  try {
    const task = await invokeCmd('get_arc_task', { task_id: taskId });
    arcDebugProgram = [];
    arcDebugStepIndex = 0;
    arcDebugCurrentGrid = null;
    const pairSel = document.getElementById('arc-debug-pair');
    pairSel.innerHTML = '';
    task.train_pairs.forEach((_, i) => {
      const opt = document.createElement('option');
      opt.value = `train-${i}`;
      opt.textContent = `Train ${i}`;
      pairSel.appendChild(opt);
    });
    if (task.test_input.length > 0) {
      const opt = document.createElement('option');
      opt.value = `test-0`;
      opt.textContent = `Test 0`;
      pairSel.appendChild(opt);
    }
    // Cache task for pair loading
    window._arcDebugTask = task;
    arcDebugLoadPair();
  } catch (e) {
    console.error('Failed to load task:', e);
  }
};

window.arcDebugLoadPair = function() {
  const task = window._arcDebugTask;
  if (!task) return;
  const pairVal = document.getElementById('arc-debug-pair').value;
  let input, expected;
  if (pairVal.startsWith('train-')) {
    const idx = parseInt(pairVal.split('-')[1]);
    input = task.train_pairs[idx].input;
    expected = task.train_pairs[idx].output;
  } else {
    input = task.test_input;
    expected = task.test_output || null;
  }
  arcDebugCurrentGrid = input.map(r => [...r]);
  arcDebugStepIndex = 0;
  arcDebugProgram = [];
  arcDebugRenderGrid('arc-debug-input', input, task.width, task.height);
  arcDebugRenderGrid('arc-debug-output', input, task.width, task.height);
  if (expected) {
    arcDebugRenderGrid('arc-debug-expected', expected, task.width, task.height);
  } else {
    document.getElementById('arc-debug-expected').innerHTML = '<div style="color:#666;padding:20px;">No expected output</div>';
  }
  arcDebugRenderProgram();
  arcDebugCalcAccuracy(input, expected);
  arcDebugLog(`Loaded ${task.name}: ${pairVal}`);
};

window.arcDebugRenderGrid = function(containerId, data, width, height) {
  const container = document.getElementById(containerId);
  if (!container || !data) return;
  container.innerHTML = '';
  container.style.gridTemplateColumns = `repeat(${width || data[0].length}, 24px)`;
  container.style.gridTemplateRows = `repeat(${height || data.length}, 24px)`;
  data.forEach(row => {
    row.forEach(val => {
      const cell = document.createElement('div');
      cell.className = `arc-cell c${val}`;
      container.appendChild(cell);
    });
  });
};

window.arcDebugUpdateParams = function() {
  const opCode = parseInt(document.getElementById('arc-debug-op').value);
  const op = ARC_OPS.find(o => o.code === opCode);
  const container = document.getElementById('arc-debug-params');
  if (!op || op.params.length === 0) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = op.params.map(p => `
    <div class="param-row">
      <span class="param-label">${p.label}</span>
      <input type="number" class="param-input" id="param-${p.key}" value="${p.val}" min="${p.min}" max="${p.max}" step="1">
    </div>
  `).join('');
};

window.arcDebugAddToken = function() {
  const opCode = parseInt(document.getElementById('arc-debug-op').value);
  const op = ARC_OPS.find(o => o.code === opCode);
  if (!op) return;
  const params = new Array(7).fill(0);
  if (op.params) {
    op.params.forEach((p, i) => {
      const el = document.getElementById(`param-${p.key}`);
      if (el) params[i] = parseInt(el.value) || 0;
    });
  }
  const token = [opCode, ...params];
  arcDebugProgram.push(token);
  arcDebugRenderProgram();
  arcDebugLog(`Added token #${arcDebugProgram.length - 1}: ${op.name} [${params.join(',')}]`);
};

window.arcDebugStep = async function() {
  if (arcDebugStepIndex >= arcDebugProgram.length) {
    arcDebugLog('Program complete');
    return;
  }
  const token = arcDebugProgram[arcDebugStepIndex];
  try {
    const result = await invokeCmd('apply_arc_token', {
      grid: arcDebugCurrentGrid,
      token_bytes: token
    });
    arcDebugCurrentGrid = result;
    arcDebugStepIndex++;
    const task = window._arcDebugTask;
    const width = task ? task.width : (arcDebugCurrentGrid[0] ? arcDebugCurrentGrid[0].length : 10);
    const height = arcDebugCurrentGrid.length;
    arcDebugRenderGrid('arc-debug-output', arcDebugCurrentGrid, width, height);
    arcDebugRenderProgram();
    // Calculate accuracy against expected
    const pairVal = document.getElementById('arc-debug-pair').value;
    let expected = null;
    if (pairVal.startsWith('train-') && task) {
      const idx = parseInt(pairVal.split('-')[1]);
      expected = task.train_pairs[idx].output;
    } else if (pairVal === 'test-0' && task && task.test_output) {
      expected = task.test_output;
    }
    arcDebugCalcAccuracy(arcDebugCurrentGrid, expected);
    const op = ARC_OPS.find(o => o.code === token[0]);
    arcDebugLog(`Step ${arcDebugStepIndex}: ${op ? op.name : 'Unknown'} applied`);
  } catch (e) {
    arcDebugLog(`Step failed: ${e}`);
  }
};

window.arcDebugReset = function() {
  arcDebugStepIndex = 0;
  arcDebugCurrentGrid = null;
  arcDebugLoadPair();
  arcDebugLog('Reset to input');
};

window.arcDebugPlay = async function() {
  if (arcDebugPlaying) return;
  arcDebugPlaying = true;
  while (arcDebugStepIndex < arcDebugProgram.length && arcDebugPlaying) {
    await arcDebugStep();
    await new Promise(r => setTimeout(r, 300));
  }
  arcDebugPlaying = false;
};

window.arcDebugClear = function() {
  arcDebugPlaying = false;
  arcDebugProgram = [];
  arcDebugStepIndex = 0;
  arcDebugRenderProgram();
  arcDebugLog('Program cleared');
};

window.arcDebugRemoveToken = function(index) {
  arcDebugProgram.splice(index, 1);
  arcDebugRenderProgram();
  arcDebugLog(`Removed token #${index}`);
};

window.arcDebugRenderProgram = function() {
  const container = document.getElementById('arc-debug-program');
  if (!container) return;
  container.innerHTML = '';
  if (arcDebugProgram.length === 0) {
    container.innerHTML = '<div style="color:#444;font-size:11px;">No tokens yet</div>';
    return;
  }
  arcDebugProgram.forEach((token, i) => {
    const op = ARC_OPS.find(o => o.code === token[0]);
    const name = op ? op.name : 'Unknown';
    const params = token.slice(1).filter(p => p !== 0);
    const chip = document.createElement('div');
    chip.className = 'arc-debug-token';
    chip.innerHTML = `<span class="token-idx">#${i}</span> ${name}${params.length ? ' ['+params.join(',')+']' : ''} <span class="token-remove" onclick="arcDebugRemoveToken(${i})">×</span>`;
    container.appendChild(chip);
  });
};

window.arcDebugCalcAccuracy = function(current, expected) {
  if (!current || !expected) {
    document.getElementById('accuracy-fill').style.width = '0%';
    document.getElementById('accuracy-value').textContent = '0%';
    return;
  }
  let total = 0, match = 0;
  const rows = Math.min(current.length, expected.length);
  for (let r = 0; r < rows; r++) {
    const cols = Math.min(current[r].length, expected[r].length);
    for (let c = 0; c < cols; c++) {
      total++;
      if (current[r][c] === expected[r][c]) match++;
    }
  }
  const pct = total > 0 ? Math.round((match / total) * 100) : 0;
  document.getElementById('accuracy-fill').style.width = pct + '%';
  document.getElementById('accuracy-value').textContent = pct + '%';
};

window.arcDebugLog = function(msg) {
  const log = document.getElementById('arc-debug-log');
  if (!log) return;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-t">[${t}]</span> <span>${msg}</span>`;
  log.insertBefore(line, log.firstChild);
  if (log.children.length > 50) log.lastChild.remove();
};

window.arcDebugRunBenchmark = async function() {
  const path = document.getElementById('bench-path').value;
  const depth = parseInt(document.getElementById('bench-depth').value) || 3;
  const resultDiv = document.getElementById('bench-result');
  resultDiv.style.display = 'block';
  resultDiv.innerHTML = '<div class="bench-header">Running benchmark...</div>';
  arcDebugLog(`Starting benchmark: ${path}, depth=${depth}`);
  try {
    const result = await invokeCmd('run_arc_benchmark', {
      dataset_path: path,
      max_depth: depth
    });
    arcDebugRenderBenchmark(result);
    arcDebugLog(`Benchmark complete: ${result.solved}/${result.total} solved (${result.accuracy_pct.toFixed(1)}%)`);
  } catch (e) {
    resultDiv.innerHTML = `<div style="color:#ff4136;">Error: ${e}</div>`;
    arcDebugLog(`Benchmark failed: ${e}`);
  }
};

window.arcDebugRenderBenchmark = function(result) {
  const resultDiv = document.getElementById('bench-result');
  if (!resultDiv) return;
  const depthLabels = ['Depth 1', 'Depth 2', 'Depth 3', 'Depth 4+'];
  const depthHtml = result.depth_distribution.map((count, i) => {
    const pct = result.total > 0 ? ((count as f64 / result.total as f64) * 100.0).toFixed(1) : 0.0;
    return `<div class="bench-stat"><span class="bench-label">${depthLabels[i]}</span><span class="bench-value">${count} (${pct}%)</span></div>`;
  }).join('');
  resultDiv.innerHTML = `
    <div class="bench-header">BENCHMARK RESULTS</div>
    <div class="bench-stat"><span class="bench-label">Total</span><span class="bench-value">${result.total}</span></div>
    <div class="bench-stat"><span class="bench-label">Solved</span><span class="bench-value">${result.solved} (${result.accuracy_pct.toFixed(1)}%)</span></div>
    <div class="bench-stat"><span class="bench-label">Failed</span><span class="bench-value">${result.failed}</span></div>
    <div class="bench-stat"><span class="bench-label">Total time</span><span class="bench-value">${(result.total_time_ms / 1000.0).toFixed(2)}s</span></div>
    <div class="bench-stat"><span class="bench-label">Avg time</span><span class="bench-value">${result.avg_time_ms.toFixed(0)}ms</span></div>
    <div class="bench-bar"><div class="bench-fill" style="width:${result.accuracy_pct.toFixed(1)}%"></div></div>
    <div style="margin-top:8px;color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Depth distribution</div>
    ${depthHtml}
  `;
};

// Hook tab switch
const origSwitchTab = window.switchTab;
window.switchTab = function(name) {
  origSwitchTab(name);
  if (name === 'arc-debug') arcDebugInit();
};


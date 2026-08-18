use goldworm::{
    vision::dsl_solver::find_solving_program,
    vision::{ArcGrid, ArcTask, ArcDataset},
    arc_apply::apply_arc_op,
    arc_program::ArcOpToken,
    arc_search::{search_program, SearchConfig},
};
use goldworm::swarm::snn_core::{SnnCore, TOTAL_NEURONS};
use rand::Rng;
use goldworm::routing::datatype_universal::{DataType, encode_datatype, decode_datatype, data_type_tag, type_tag_name};
use serde::{Serialize, Deserialize};
use std::time::{Instant, Duration};

// ============================================================================
// SNN Commands
// ============================================================================

#[tauri::command]
pub fn init_snn_core(density: f64) -> Result<goldworm::swarm::snn_core::SnnStateDto, String> {
    let core = SnnCore::new(density);
    Ok(goldworm::swarm::snn_core::SnnStateDto::from(&core))
}

#[tauri::command]
pub fn step_snn(state: goldworm::swarm::snn_core::SnnStateDto, input_spikes: Vec<u32>) -> Result<goldworm::swarm::snn_core::SnnStateDto, String> {
    let mut core = SnnCore::new(state.density);
    for n in &state.neurons {
        if n.id < TOTAL_NEURONS {
            core.swarm.arena.membrane[n.id] = n.v_m;
            core.swarm.arena.refractory[n.id] = n.refractory as u32;
        }
    }
    core.tick = state.tick;
    let input_spikes_usize: Vec<usize> = input_spikes.into_iter().map(|x| x as usize).collect();
    core.step(&input_spikes_usize);
    Ok(goldworm::swarm::snn_core::SnnStateDto::from(&core))
}

// ============================================================================
// ARC Command
// ============================================================================

fn flat_to_grid(flat: &[u8]) -> ArcGrid {
    let mut data = vec![vec![0u8; 10]; 10];
    for (i, &val) in flat.iter().enumerate() {
        if i < 100 {
            let r = i / 10;
            let c = i % 10;
            data[r][c] = val;
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn grid_to_flat(grid: &ArcGrid) -> Vec<u8> {
    grid.data.iter().flatten().cloned().collect()
}

#[tauri::command]
pub fn solve_arc_task(input_grid: Vec<u8>, _operation: String) -> Result<Vec<u8>, String> {
    let grid = flat_to_grid(&input_grid);
    let expected = grid.clone();
    let task = ArcTask {
        id: "tauri_task".to_string(),
        train_pairs: vec![(grid.clone(), expected)],
        test_inputs: vec![grid.clone()],
        test_outputs: vec![None],
    };
    let program = find_solving_program(&task, 3).ok_or("No solving program found")?;
    let output = program.apply(&grid).ok_or("Program application failed")?;
    Ok(grid_to_flat(&output))
}

// ============================================================================
// ARC Debugger Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcPairDto {
    pub input: Vec<Vec<u8>>,
    pub output: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcTaskDto {
    pub id: String,
    pub name: String,
    pub description: String,
    pub train_pairs: Vec<ArcPairDto>,
    pub test_input: Vec<Vec<u8>>,
    pub test_output: Option<Vec<Vec<u8>>>,
    pub width: usize,
    pub height: usize,
}

fn make_grid(data: Vec<Vec<u8>>) -> ArcGrid {
    ArcGrid::from_data(data).unwrap()
}

fn mock_tasks() -> Vec<ArcTaskDto> {
    vec![
        ArcTaskDto {
            id: "move_right".into(),
            name: "Move Right".into(),
            description: "Move a shape 1 cell to the right.".into(),
            width: 3,
            height: 3,
            train_pairs: vec![
                ArcPairDto {
                    input: vec![vec![1,0,0], vec![0,0,0], vec![0,0,0]],
                    output: vec![vec![0,1,0], vec![0,0,0], vec![0,0,0]],
                },
                ArcPairDto {
                    input: vec![vec![0,0,2], vec![0,0,0], vec![0,0,0]],
                    output: vec![vec![0,0,0], vec![0,0,2], vec![0,0,0]],
                },
            ],
            test_input: vec![vec![0,0,0], vec![3,0,0], vec![0,0,0]],
            test_output: Some(vec![vec![0,0,0], vec![0,3,0], vec![0,0,0]]),
        },
        ArcTaskDto {
            id: "fill_rect".into(),
            name: "Fill Rectangle".into(),
            description: "Fill a 2x2 rectangle with color 5.".into(),
            width: 4,
            height: 4,
            train_pairs: vec![
                ArcPairDto {
                    input: vec![vec![0,0,0,0], vec![0,1,1,0], vec![0,1,1,0], vec![0,0,0,0]],
                    output: vec![vec![0,0,0,0], vec![0,5,5,0], vec![0,5,5,0], vec![0,0,0,0]],
                },
            ],
            test_input: vec![vec![0,0,0,0], vec![0,2,2,0], vec![0,2,2,0], vec![0,0,0,0]],
            test_output: Some(vec![vec![0,0,0,0], vec![0,5,5,0], vec![0,5,5,0], vec![0,0,0,0]]),
        },
        ArcTaskDto {
            id: "gravity_down".into(),
            name: "Gravity Down".into(),
            description: "Let pixels fall to the bottom.".into(),
            width: 3,
            height: 3,
            train_pairs: vec![
                ArcPairDto {
                    input: vec![vec![1,0,0], vec![0,2,0], vec![0,0,3]],
                    output: vec![vec![0,0,0], vec![0,0,0], vec![1,2,3]],
                },
            ],
            test_input: vec![vec![0,4,0], vec![0,0,0], vec![5,0,6]],
            test_output: Some(vec![vec![0,0,0], vec![0,0,0], vec![5,4,6]]),
        },
        ArcTaskDto {
            id: "flip_h".into(),
            name: "Flip Horizontal".into(),
            description: "Mirror the grid left-right.".into(),
            width: 4,
            height: 3,
            train_pairs: vec![
                ArcPairDto {
                    input: vec![vec![1,2,3,4], vec![5,6,7,8], vec![0,0,0,0]],
                    output: vec![vec![4,3,2,1], vec![8,7,6,5], vec![0,0,0,0]],
                },
            ],
            test_input: vec![vec![9,0,0,9], vec![0,1,1,0], vec![0,0,0,0]],
            test_output: Some(vec![vec![9,0,0,9], vec![0,1,1,0], vec![0,0,0,0]]),
        },
    ]
}

#[tauri::command]
pub fn list_arc_tasks() -> Vec<String> {
    mock_tasks().into_iter().map(|t| t.id).collect()
}

#[tauri::command]
pub fn get_arc_task(task_id: String) -> Result<ArcTaskDto, String> {
    mock_tasks().into_iter().find(|t| t.id == task_id).ok_or_else(|| format!("Task not found: {}", task_id))
}

#[tauri::command]
pub fn apply_arc_token(grid: Vec<Vec<u8>>, token_bytes: [u8; 8]) -> Result<Vec<Vec<u8>>, String> {
    let grid = make_grid(grid);
    let token = ArcOpToken(token_bytes);
    let result = apply_arc_op(&grid, &token).ok_or("Operation failed (out-of-bounds or invalid)")?;
    Ok(result.data)
}

// ============================================================================
// ARC Benchmark Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcBenchmarkTaskResult {
    pub task_id: String,
    pub solved: bool,
    pub program_length: Option<usize>,
    pub candidates: usize,
    pub time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcBenchmarkResult {
    pub total: usize,
    pub solved: usize,
    pub failed: usize,
    pub accuracy_pct: f64,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    pub depth_distribution: Vec<usize>,
    pub tasks: Vec<ArcBenchmarkTaskResult>,
}

#[tauri::command]
pub fn run_arc_benchmark(dataset_path: String, max_depth: usize) -> Result<ArcBenchmarkResult, String> {
    let dir = std::path::PathBuf::from(dataset_path);
    let dataset = ArcDataset::load_from_directory(dir).map_err(|e| e.to_string())?;
    
    let total = dataset.tasks.len();
    let mut solved = 0;
    let mut failed = 0;
    let mut depth_distribution = vec![0usize; 4]; // depth 1-3, 4+
    let start = Instant::now();
    let mut tasks = Vec::with_capacity(total);
    
    for (i, task) in dataset.tasks.iter().enumerate() {
        let task_start = Instant::now();
        let config = SearchConfig {
            max_depth,
            ..Default::default()
        };
        let result = search_program(task, config);
        let time_ms = task_start.elapsed().as_secs_f64() * 1000.0;
        
        if let Some(ref prog) = result.program {
            solved += 1;
            let depth = prog.len();
            if depth <= 3 {
                depth_distribution[depth - 1] += 1;
            } else {
                depth_distribution[3] += 1;
            }
            tasks.push(ArcBenchmarkTaskResult {
                task_id: task.id.clone(),
                solved: true,
                program_length: Some(depth),
                candidates: result.candidates_evaluated,
                time_ms,
            });
        } else {
            failed += 1;
            tasks.push(ArcBenchmarkTaskResult {
                task_id: task.id.clone(),
                solved: false,
                program_length: None,
                candidates: result.candidates_evaluated,
                time_ms,
            });
        }
    }
    
    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let avg_time = if total > 0 { total_time / total as f64 } else { 0.0 };
    let accuracy_pct = if total > 0 { (solved as f64 / total as f64) * 100.0 } else { 0.0 };
    
    Ok(ArcBenchmarkResult {
        total,
        solved,
        failed,
        accuracy_pct,
        total_time_ms: total_time,
        avg_time_ms: avg_time,
        depth_distribution,
        tasks,
    })
}

// ============================================================================
// Utility Commands
// ============================================================================

#[tauri::command]
pub fn get_monster_points() -> Vec<[f64; 3]> {
    let hardcoded: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [-0.110605, 0.101324, 1.198661],
        [0.018546, -0.21132, -0.113202],
        [0.158077, 0.206183, -1.187971],
        [-0.295414, -0.052254, 0.225394],
        [0.283004, -0.180024, 1.166684],
        [-0.095384, 0.354827, -0.335576],
        [-0.182917, -0.352195, -1.134993],
        [0.398521, 0.145538, 0.442765],
        [-0.415955, 0.171701, 1.093178],
        [0.201047, -0.429628, -0.546005],
        [0.148893, 0.47469, -1.041613],
        [-0.449578, -0.260538, 0.644375],
    ];

    let mut points = hardcoded.clone();
    let mut rng = rand::thread_rng();
    for i in points.len()..196 {
        let t = i as f64 / 196.0 * std::f64::consts::PI * 8.0;
        points.push([
            (t.cos() * (0.5 + rng.gen::<f64>() * 1.5)),
            (t.sin() * (0.5 + rng.gen::<f64>() * 1.5)),
            (t.sin() * 1.2),
        ]);
    }
    points
}

#[tauri::command]
pub fn get_status() -> Result<String, String> {
    Ok("GoldWorm engine ready".to_string())
}

// ============================================================================
// Universal DataType Interface — SHD-CCP Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    pub tag: u8,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedPayload {
    pub type_tag: u8,
    pub type_name: String,
    pub size_bytes: usize,
    pub hex: String,
    pub base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedPayload {
    pub type_tag: u8,
    pub type_name: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEventDto {
    pub src: usize,
    pub dst: usize,
    pub delay_ticks: u16,
    pub amplitude_u8: u8,
    pub flags: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicPointDto {
    pub coords: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuaternionDto {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateArenaDto {
    pub membrane: Vec<f32>,
    pub recovery: Vec<f32>,
    pub threshold: Vec<f32>,
    pub refractory: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcGridDto {
    pub data: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DvsEventDto {
    pub x: u8,
    pub y: u8,
    pub polarity: u8,
    pub timestamp_us: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconTokenDto {
    pub id: usize,
    pub surface: String,
    pub class: String,
    pub embedding: QuaternionDto,
    pub hyperbolic: HyperbolicPointDto,
    pub salience: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMatrixDto {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[tauri::command]
pub fn list_supported_types() -> Vec<TypeInfo> {
    vec![
        TypeInfo { tag: 0x01, name: "SpikeStream".into(), description: "Vec<SpikeEvent> — delta-encoded spike trains".into() },
        TypeInfo { tag: 0x02, name: "StateArena".into(), description: "Flat neuron state (membrane, recovery, threshold, refractory)".into() },
        TypeInfo { tag: 0x03, name: "WeightMatrix".into(), description: "Row-major flat weight matrix".into() },
        TypeInfo { tag: 0x04, name: "ShdSample".into(), description: "SHD audio sample (spikes + label)".into() },
        TypeInfo { tag: 0x05, name: "ArcGrid".into(), description: "2D ARC grid (Vec<Vec<u8>>)".into() },
        TypeInfo { tag: 0x06, name: "HyperbolicPoint".into(), description: "Poincaré-ball coordinates".into() },
        TypeInfo { tag: 0x07, name: "Quaternion".into(), description: "Unit quaternion (w,x,y,z)".into() },
        TypeInfo { tag: 0x08, name: "DvsEventBatch".into(), description: "Batch of DVS128 events".into() },
        TypeInfo { tag: 0x09, name: "AvalancheMetrics".into(), description: "Telemetry snapshot".into() },
        TypeInfo { tag: 0x0A, name: "LexiconToken".into(), description: "Semantic token with embedding".into() },
        TypeInfo { tag: 0x0B, name: "SHDCCP".into(), description: "Sparse CSR matrix (SHD-CCP format)".into() },
        TypeInfo { tag: 0x0C, name: "MoAIndex".into(), description: "Mixture-of-Agents routing index".into() },
        TypeInfo { tag: 0x0D, name: "SwarmConfig".into(), description: "QLIF swarm parameters".into() },
        TypeInfo { tag: 0x0E, name: "PoincareBall".into(), description: "Hyperbolic geometry curvature".into() },
        TypeInfo { tag: 0x0F, name: "DvsEncoderConfig".into(), description: "DVS encoder settings".into() },
        TypeInfo { tag: 0x10, name: "ConversationTurn".into(), description: "Chat turn (role, content, timestamp)".into() },
    ]
}

#[tauri::command]
pub fn encode_spike_stream(events: Vec<SpikeEventDto>) -> EncodedPayload {
    let spikes: Vec<goldworm::substrate::SpikeEvent> = events.into_iter().map(|e| {
        goldworm::substrate::SpikeEvent {
            src: goldworm::substrate::NeuronIdx(e.src),
            dst: goldworm::substrate::NeuronIdx(e.dst),
            delay_ticks: e.delay_ticks,
            amplitude_u8: e.amplitude_u8,
            flags: e.flags,
        }
    }).collect();
    let dt = DataType::SpikeStream(spikes);
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_arc_grid(grid: ArcGridDto) -> EncodedPayload {
    let dt = DataType::ArcGrid(goldworm::vision::ArcGrid {
        data: grid.data,
        width: grid.width,
        height: grid.height,
    });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_hyperbolic_point(dto: HyperbolicPointDto) -> EncodedPayload {
    let dt = DataType::HyperbolicPoint(goldworm::geometry::HyperbolicPoint { coords: dto.coords });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_quaternion(dto: QuaternionDto) -> EncodedPayload {
    let dt = DataType::Quaternion(goldworm::geometry::Quaternion {
        w: dto.w, x: dto.x, y: dto.y, z: dto.z,
    });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_state_arena(dto: StateArenaDto) -> EncodedPayload {
    let dt = DataType::StateArena(goldworm::substrate::StateArena { membrane: dto.membrane, recovery: dto.recovery, threshold: dto.threshold, refractory: dto.refractory });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_weight_matrix(dto: WeightMatrixDto) -> EncodedPayload {
    let dt = DataType::WeightMatrix(goldworm::substrate::WeightMatrix {
        data: dto.data,
        rows: dto.rows,
        cols: dto.cols,
    });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_dvs_batch(events: Vec<DvsEventDto>) -> EncodedPayload {
    let evts: Vec<goldworm::chat::dvs_encoder::DvsEvent> = events.into_iter().map(|e| {
        goldworm::chat::dvs_encoder::DvsEvent::new(e.x, e.y, e.polarity, e.timestamp_us)
    }).collect();
    let dt = DataType::DvsEventBatch(evts);
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn encode_lexicon_token(dto: LexiconTokenDto) -> EncodedPayload {
    let dt = DataType::LexiconToken(goldworm::semantics::token_engine::LexiconToken {
        id: dto.id,
        surface: dto.surface,
        class: class_from_str(&dto.class),
        embedding: goldworm::geometry::Quaternion {
            w: dto.embedding.w,
            x: dto.embedding.x,
            y: dto.embedding.y,
            z: dto.embedding.z,
        },
        hyperbolic: goldworm::geometry::HyperbolicPoint { coords: dto.hyperbolic.coords },
        salience: dto.salience,
    });
    encode_and_wrap(dt)
}

#[tauri::command]
pub fn decode_payload(hex: String) -> Result<DecodedPayload, String> {
    let bytes = hex_to_bytes(&hex).map_err(|e| e.to_string())?;
    let dt = decode_datatype(&bytes)?;
    let tag = data_type_tag(&dt);
    Ok(DecodedPayload {
        type_tag: tag as u8,
        type_name: type_tag_name(tag).into(),
        summary: summarize_datatype(&dt),
    })
}

#[tauri::command]
pub fn decode_to_json(hex: String) -> Result<String, String> {
    let bytes = hex_to_bytes(&hex).map_err(|e| e.to_string())?;
    let dt = decode_datatype(&bytes)?;
    serde_json::to_string(&dt_to_json(dt)).map_err(|e| e.to_string())
}

// ============================================================================
// Helpers
// ============================================================================

fn encode_and_wrap(dt: DataType) -> EncodedPayload {
    let bytes = encode_datatype(&dt);
    let tag = data_type_tag(&dt);
    let hex = bytes_to_hex(&bytes);
    let base64 = base64_encode(&bytes);
    EncodedPayload {
        type_tag: tag as u8,
        type_name: type_tag_name(tag).into(),
        size_bytes: bytes.len(),
        hex,
        base64,
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 { return Err("Odd hex length".into()); }
    (0..hex.len()).step_by(2).map(|i| {
        u8::from_str_radix(&hex[i..i+2], 16).map_err(|e| e.to_string())
    }).collect()
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(bytes)
}

fn class_from_str(s: &str) -> goldworm::semantics::token_engine::TokenClass {
    use goldworm::semantics::token_engine::TokenClass;
    match s {
        "Determiner" => TokenClass::Determiner,
        "NounConcrete" => TokenClass::NounConcrete,
        "NounAbstract" => TokenClass::NounAbstract,
        "VerbAction" => TokenClass::VerbAction,
        "VerbState" => TokenClass::VerbState,
        "Adjective" => TokenClass::Adjective,
        "Preposition" => TokenClass::Preposition,
        "SemanticRole" => TokenClass::SemanticRole,
        "GrammarMarker" => TokenClass::GrammarMarker,
        "Punctuation" => TokenClass::Punctuation,
        _ => TokenClass::Noise,
    }
}

fn summarize_datatype(dt: &DataType) -> String {
    match dt {
        DataType::SpikeStream(s) => format!("{} spike events", s.len()),
        DataType::SpikeBuffer(b) => format!("{} spike indices", b.indices.len()),
        DataType::StateArena(a) => format!("{} neurons", a.membrane.len()),
        DataType::WeightMatrix(w) => format!("{}x{} matrix, {} values", w.rows, w.cols, w.data.len()),
        DataType::ShdSample(s) => format!("{} spikes, label={}", s.spikes.len(), s.label),
        DataType::ArcGrid(g) => format!("{}x{} grid", g.width, g.height),
        DataType::HyperbolicPoint(p) => format!("dim={} point", p.coords.len()),
        DataType::Quaternion(q) => format!("quaternion norm={:.4}", q.norm()),
        DataType::DvsEventBatch(e) => format!("{} DVS events", e.len()),
        DataType::AvalancheMetrics(m) => format!("total_spikes={}", m.total_spikes),
        DataType::LexiconToken(t) => format!("token '{}' ({:?})", t.surface, t.class),
        DataType::SHDCCP(s) => format!("{} nonzeros, {} rows", s.values.len(), s.row_ptr.len() - 1),
        DataType::MoAIndex(m) => format!("{} experts", m.expert_indices.len()),
        DataType::SwarmConfig(c) => format!("decay={}, rest={}, noise={}", c.decay, c.resting_potential, c.noise_std),
        DataType::PoincareBall(p) => format!("curvature={}", p.curvature),
        DataType::DvsEncoderConfig(c) => format!("window={}us, spikes/evt={}", c.window_size_us, c.spikes_per_event),
        DataType::ConversationTurn(t) => format!("{}: {}...", t.role, &t.text[..t.text.len().min(20)]),
    }
}

fn dt_to_json(dt: DataType) -> serde_json::Value {
    use serde_json::json;
    match dt {
        DataType::SpikeStream(s) => json!({ "type": "SpikeStream", "events": s.len() }),
        DataType::SpikeBuffer(b) => json!({ "type": "SpikeBuffer", "indices": b.indices.len() }),
        DataType::StateArena(a) => json!({ "type": "StateArena", "neurons": a.membrane.len() }),
        DataType::WeightMatrix(w) => json!({ "type": "WeightMatrix", "rows": w.rows, "cols": w.cols, "data": w.data }),
        DataType::ShdSample(s) => json!({ "type": "ShdSample", "spikes": s.spikes.len(), "label": s.label }),
        DataType::ArcGrid(g) => json!({ "type": "ArcGrid", "width": g.width, "height": g.height, "data": g.data }),
        DataType::HyperbolicPoint(p) => json!({ "type": "HyperbolicPoint", "coords": p.coords }),
        DataType::Quaternion(q) => json!({ "type": "Quaternion", "w": q.w, "x": q.x, "y": q.y, "z": q.z }),
        DataType::DvsEventBatch(e) => json!({ "type": "DvsEventBatch", "events": e.len() }),
        DataType::AvalancheMetrics(m) => json!({ "type": "AvalancheMetrics", "total_spikes": m.total_spikes }),
        DataType::LexiconToken(t) => json!({ "type": "LexiconToken", "surface": t.surface, "class": format!("{:?}", t.class) }),
        DataType::SHDCCP(s) => json!({ "type": "SHDCCP", "values": s.values.len(), "rows": s.row_ptr.len() }),
        DataType::MoAIndex(m) => json!({ "type": "MoAIndex", "experts": m.expert_indices.len() }),
        DataType::SwarmConfig(c) => json!({ "type": "SwarmConfig", "decay": c.decay, "resting_potential": c.resting_potential, "noise_std": c.noise_std }),
        DataType::PoincareBall(p) => json!({ "type": "PoincareBall", "curvature": p.curvature }),
        DataType::DvsEncoderConfig(c) => json!({ "type": "DvsEncoderConfig", "window_size_us": c.window_size_us }),
        DataType::ConversationTurn(t) => json!({ "type": "ConversationTurn", "role": t.role, "content": t.text }),
    }
}

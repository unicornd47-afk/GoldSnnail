use std::fs;

fn main() {
    let path = "data/nmnis_t/Train/3/34290.bin";
    let data = fs::read(path).unwrap();
    let num_events = data.len() / 5;
    println!("File: {}", path);
    println!("File size: {} bytes", data.len());
    println!("Number of 5-byte events: {}", num_events);
    println!("File size / 8 = {}", data.len() / 8);
    println!();

    // Print first 10 events with 5-byte interpretation
    println!("=== 5-byte events (correct format) ===");
    for i in 0..10.min(num_events) {
        let off = i * 5;
        let b0 = data[off];
        let b1 = data[off + 1];
        let b2 = data[off + 2];
        let b3 = data[off + 3];
        let b4 = data[off + 4];

        // Try little-endian 4-byte word
        let raw_le = u32::from_le_bytes([b0, b1, b2, b3]);
        let x_le = (raw_le & 0x3F) as u8;
        let y_le = ((raw_le >> 6) & 0x3F) as u8;
        let pol_le = ((raw_le >> 12) & 0x1) as u8;
        let ts_low_le = raw_le >> 13;
        let ts_le = ts_low_le | ((b4 as u32) << 19);

        // Try big-endian 4-byte word
        let raw_be = u32::from_be_bytes([b0, b1, b2, b3]);
        let x_be = ((raw_be >> 26) & 0x3F) as u8;
        let y_be = ((raw_be >> 20) & 0x3F) as u8;
        let pol_be = ((raw_be >> 19) & 0x1) as u8;
        let ts_low_be = raw_be & 0x7FFFF;
        let ts_be = ts_low_be | ((b4 as u32) << 19);

        println!("Event {}: bytes = [{:3} {:3} {:3} {:3} {:3}]", i, b0, b1, b2, b3, b4);
        println!("  LE: x={}, y={}, pol={}, ts={}", x_le, y_le, pol_le, ts_le);
        println!("  BE: x={}, y={}, pol={}, ts={}", x_be, y_be, pol_be, ts_be);
    }

    // Check timestamp range with LE interpretation
    let mut min_ts_le = u32::MAX;
    let mut max_ts_le = 0u32;
    let mut min_ts_be = u32::MAX;
    let mut max_ts_be = 0u32;
    for i in 0..num_events {
        let off = i * 5;
        let raw_le = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let ts_le = (raw_le >> 13) | ((data[off+4] as u32) << 19);
        let raw_be = u32::from_be_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let ts_be = (raw_be & 0x7FFFF) | ((data[off+4] as u32) << 19);
        min_ts_le = min_ts_le.min(ts_le);
        max_ts_le = max_ts_le.max(ts_le);
        min_ts_be = min_ts_be.min(ts_be);
        max_ts_be = max_ts_be.max(ts_be);
    }
    println!();
    println!("LE timestamp range: {} - {} (span: {} us)", min_ts_le, max_ts_le, max_ts_le - min_ts_le);
    println!("BE timestamp range: {} - {} (span: {} us)", min_ts_be, max_ts_be, max_ts_be - min_ts_be);

    // Check x, y ranges with LE
    let mut xs = vec![];
    let mut ys = vec![];
    let mut pols = vec![];
    for i in 0..num_events {
        let off = i * 5;
        let raw_le = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        xs.push((raw_le & 0x3F) as u8);
        ys.push(((raw_le >> 6) & 0x3F) as u8);
        pols.push(((raw_le >> 12) & 0x1) as u8);
    }
    xs.sort(); ys.sort(); pols.sort();
    println!();
    println!("LE x range: {} - {}", xs[0], xs[xs.len()-1]);
    println!("LE y range: {} - {}", ys[0], ys[ys.len()-1]);
    let pol0 = pols.iter().filter(|&&p| p == 0).count();
    let pol1 = pols.iter().filter(|&&p| p == 1).count();
    println!("LE polarity: 0={}, 1={}", pol0, pol1);
}

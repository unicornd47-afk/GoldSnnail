import sys
with open('src/harness/arc_tripartite.rs', 'r', encoding='utf-8') as f:
    lines = f.readlines()
# Find the end of mod tests block: look for the last "}" before any non-indented content
truncate_point = len(lines)
for i in range(len(lines)-1, -1, -1):
    stripped = lines[i].strip()
    if stripped == '}':
        truncate_point = i + 1
        break

new_content = ''.join(lines[:truncate_point])
new_content += '''
    #[test]
    fn test_scale_width_spatial_identity() {
        let enc1 = ArcTripartiteEncoder::new(10, 10, 1);
        let enc2 = ArcTripartiteEncoder::new(10, 10, 2);
        let code1 = enc1.encode(5, 3, 3, ArcPhase::DemoInput);
        let code2 = enc2.encode(5, 3, 3, ArcPhase::DemoInput);
        
        // Extract spatial blocks: indices 10..70 for both
        let spatial1 = &code1[10..70];
        let spatial2 = &code2[10..70];
        
        for i in 0..60 {
            assert!(
                (spatial1[i] - spatial2[i]).abs() < 1e-5,
                "Spatial dim {} mismatch: {} vs {}",
                i, spatial1[i], spatial2[i]
            );
        }
    }
}
'''
with open('src/harness/arc_tripartite.rs', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('File fixed')

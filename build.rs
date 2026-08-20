use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/");
    
    let shaders = ["qlif_step.comp", "hyperbolic.comp", "exp_map.comp"];
    
    for shader in &shaders {
        let input = format!("shaders/{}", shader);
        let output = format!("shaders/{}.spv", shader);
        
        let status = Command::new("glslangValidator")
            .args(&["-V", &input, "-o", &output])
            .status();
            
        if status.is_err() || !status.unwrap().success() {
            // Fallback: shaderc via cargo
            let _ = Command::new("cargo")
                .args(&["run", "-p", "shaderc", "--bin", "shader_compile"])
                .status();
        }
    }
}

//! Transformation-Vektor-Analyse im Hyperbolic Space
//!
//! Testet, ob ähnliche ARC-Transformationen konsistente Vektoren
//! im Poincaré-Ball erzeugen.

use crate::geometry::PoincareBall;
use crate::vision::{ArcGrid, ArcTask, GridEncoder};

/// Eine Transformation als Hyperbolic-Vektor (Output - Input).
#[derive(Debug, Clone)]
pub struct TransformationVector {
    pub task_id: String,
    pub transformation_type: String,
    pub vector: Vec<f64>,
    pub input_radius: f64,
    pub output_radius: f64,
    pub distance: f64,
}

impl TransformationVector {
    /// Cosine Similarity zwischen zwei Transformationsvektoren.
    pub fn cosine_similarity(&self, other: &TransformationVector) -> f64 {
        let dot: f64 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();
        let norm_a = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = other.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (norm_a * norm_b).max(1e-12)
    }
}

/// Analyse-Ergebnis für einen Transformationstyp.
#[derive(Debug)]
pub struct TransformationAnalysis {
    pub transformation_type: String,
    pub task_count: usize,
    pub mean_similarity: f64,
    pub std_similarity: f64,
    pub is_consistent: bool,
}

/// Analyzer für Transformationskonsistenz im Hyperbolic Space.
pub struct TransformationAnalyzer<'a> {
    pub encoder: &'a GridEncoder,
    pub ball: PoincareBall,
}

impl<'a> TransformationAnalyzer<'a> {
    pub fn new(encoder: &'a GridEncoder) -> Self {
        Self {
            encoder,
            ball: PoincareBall::new(1.0),
        }
    }

    /// Berechnet Transformationsvektor für ein Train-Paar.
    pub fn compute_vector(
        &self,
        task_id: &str,
        input: &ArcGrid,
        output: &ArcGrid,
        transformation_type: &str,
    ) -> Result<TransformationVector, String> {
        let in_point = self.encoder.encode(input)?;
        let out_point = self.encoder.encode(output)?;

        let vector: Vec<f64> = out_point
            .coords
            .iter()
            .zip(&in_point.coords)
            .map(|(a, b)| a - b)
            .collect();

        // Euclidean distance (matches PoincareBall::distance implementation)
        let distance = in_point
            .coords
            .iter()
            .zip(&out_point.coords)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(TransformationVector {
            task_id: task_id.to_string(),
            transformation_type: transformation_type.to_string(),
            vector,
            input_radius: in_point.euclidean_norm(),
            output_radius: out_point.euclidean_norm(),
            distance,
        })
    }

    /// Analysiert Konsistenz eines Transformationstyps über mehrere Tasks.
    pub fn analyze_type(
        &self,
        vectors: &[TransformationVector],
        transformation_type: &str,
    ) -> TransformationAnalysis {
        let filtered: Vec<_> = vectors
            .iter()
            .filter(|v| v.transformation_type == transformation_type)
            .collect();

        if filtered.len() < 2 {
            return TransformationAnalysis {
                transformation_type: transformation_type.to_string(),
                task_count: filtered.len(),
                mean_similarity: 0.0,
                std_similarity: 0.0,
                is_consistent: false,
            };
        }

        let mut similarities = Vec::new();
        for i in 0..filtered.len() {
            for j in (i + 1)..filtered.len() {
                similarities.push(filtered[i].cosine_similarity(filtered[j]));
            }
        }

        let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let variance = similarities.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / similarities.len() as f64;
        let std = variance.sqrt();

        let is_consistent = mean > 0.6 && std < 0.3;

        TransformationAnalysis {
            transformation_type: transformation_type.to_string(),
            task_count: filtered.len(),
            mean_similarity: mean,
            std_similarity: std,
            is_consistent,
        }
    }
}

// === Transformation Classification ===

impl GridEncoder {
    /// Erkennt Transformationstyp aus Input/Output-Grids (Heuristik).
    pub fn classify_transformation(input: &ArcGrid, output: &ArcGrid) -> &'static str {
        // Rotation 90°: transponierte Größe
        if input.width == output.height && input.height == output.width && !input.data.is_empty() {
            let mut rotated = vec![vec![0u8; output.width]; output.height];
            for r in 0..input.height {
                for c in 0..input.width {
                    rotated[c][input.height - 1 - r] = input.data[r][c];
                }
            }
            if rotated == output.data {
                return "rotation_90";
            }
        }

        // Spiegelung horizontal: Zeilen umgekehrt
        if input.width == output.width
            && input.height == output.height
            && !input.data.is_empty()
        {
            let mut flipped = input.data.clone();
            flipped.reverse();
            if flipped == output.data {
                return "flip_horizontal";
            }

            // Spiegelung vertikal: jede Zeile umgekehrt
            let mut v_flipped = input.data.clone();
            for row in &mut v_flipped {
                row.reverse();
            }
            if v_flipped == output.data {
                return "flip_vertical";
            }
        }

        // Farb-Mapping: konsistente Zuordnung
        if input.width == output.width
            && input.height == output.height
            && !input.data.is_empty()
        {
            let mut mapping = [255u8; 10];
            let mut consistent = true;
            for r in 0..input.height {
                for c in 0..input.width {
                    let in_color = input.data[r][c];
                    let out_color = output.data[r][c];
                    if mapping[in_color as usize] == 255 {
                        mapping[in_color as usize] = out_color;
                    } else if mapping[in_color as usize] != out_color {
                        consistent = false;
                        break;
                    }
                }
                if !consistent {
                    break;
                }
            }
            if consistent && mapping.iter().any(|&m| m != 255) {
                return "color_mapping";
            }
        }

        // Objekt-Zählen: Output ist einzelne Zelle mit Zahl
        if output.width == 1 && output.height == 1 {
            return "count_objects";
        }

        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v1 = TransformationVector {
            task_id: "t1".into(),
            transformation_type: "rotation_90".into(),
            vector: vec![1.0, 0.0, 0.0],
            input_radius: 0.5,
            output_radius: 0.5,
            distance: 0.1,
        };
        let v2 = TransformationVector {
            task_id: "t2".into(),
            transformation_type: "rotation_90".into(),
            vector: vec![1.0, 0.0, 0.0],
            input_radius: 0.5,
            output_radius: 0.5,
            distance: 0.1,
        };
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = TransformationVector {
            task_id: "t1".into(),
            transformation_type: "rotation_90".into(),
            vector: vec![1.0, 0.0],
            input_radius: 0.5,
            output_radius: 0.5,
            distance: 0.1,
        };
        let v2 = TransformationVector {
            task_id: "t2".into(),
            transformation_type: "rotation_90".into(),
            vector: vec![0.0, 1.0],
            input_radius: 0.5,
            output_radius: 0.5,
            distance: 0.1,
        };
        assert!((v1.cosine_similarity(&v2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_type_insufficient() {
        let encoder = GridEncoder::new(100, 32, 16, 0.75);
        let analyzer = TransformationAnalyzer::new(&encoder);
        let vectors = vec![];
        let analysis = analyzer.analyze_type(&vectors, "rotation_90");
        assert_eq!(analysis.task_count, 0);
        assert!(!analysis.is_consistent);
    }

    #[test]
    fn test_classify_rotation_90() {
        let input = ArcGrid::from_data(vec![
            vec![1, 0],
            vec![0, 2],
        ]).unwrap();
        let output = ArcGrid::from_data(vec![
            vec![0, 1],
            vec![2, 0],
        ]).unwrap();
        assert_eq!(GridEncoder::classify_transformation(&input, &output), "rotation_90");
    }

    #[test]
    fn test_classify_flip_horizontal() {
        let input = ArcGrid::from_data(vec![
            vec![1, 0],
            vec![0, 2],
        ]).unwrap();
        let output = ArcGrid::from_data(vec![
            vec![0, 2],
            vec![1, 0],
        ]).unwrap();
        assert_eq!(
            GridEncoder::classify_transformation(&input, &output),
            "flip_horizontal"
        );
    }

    #[test]
    fn test_classify_color_mapping() {
        let input = ArcGrid::from_data(vec![
            vec![1, 1],
            vec![2, 2],
        ]).unwrap();
        let output = ArcGrid::from_data(vec![
            vec![3, 3],
            vec![4, 4],
        ]).unwrap();
        assert_eq!(
            GridEncoder::classify_transformation(&input, &output),
            "color_mapping"
        );
    }

    #[test]
    fn test_classify_count_objects() {
        let input = ArcGrid::from_data(vec![
            vec![1, 0],
            vec![0, 2],
        ]).unwrap();
        let output = ArcGrid::from_data(vec![vec![2]]).unwrap();
        assert_eq!(
            GridEncoder::classify_transformation(&input, &output),
            "count_objects"
        );
    }

    #[test]
    fn test_classify_unknown() {
        let input = ArcGrid::from_data(vec![vec![1, 1], vec![2, 2]]).unwrap();
        let output = ArcGrid::from_data(vec![vec![1, 2], vec![1, 2]]).unwrap();
        assert_eq!(
            GridEncoder::classify_transformation(&input, &output),
            "unknown"
        );
    }
}

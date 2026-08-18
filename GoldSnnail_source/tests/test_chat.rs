use goldworm::{
    TokenSpikeEncoder, SpikeTokenDecoder, SpikeBuffer,
    ConversationBuffer, ConversationTurn, SemanticTrainer,
    TransitionalLearner, LexiconToken, TokenClass,
    Quaternion, HyperbolicPoint,
};
use ndarray::array;

#[test]
fn chat_cli_response_generation_smoke_test() {
    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);

    let words = vec!["hallo", "gut", "schlecht", "hund", "läuft"];
    for (i, word) in words.iter().enumerate() {
        trainer.lexicon.tokens.push(LexiconToken {
            id: i,
            surface: word.to_string(),
            class: TokenClass::NounConcrete,
            embedding: Quaternion::new(0.1, 0.1, 0.0, 0.0).normalize(),
            hyperbolic: HyperbolicPoint::new(array![0.09, 0.09]).unwrap(),
            salience: 0.5,
        });
        trainer.lexicon.word_index.insert(word.to_string(), i);
        trainer.lexicon.class_index.entry(TokenClass::NounConcrete).or_default().push(i);
    }

    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    let sentence = vec!["hallo".to_string(), "hund".to_string(), "läuft".to_string()];
    let spikes = encoder.encode_sentence(&sentence);
    assert!(!spikes.is_empty());

    let buffer = SpikeBuffer {
        indices: spikes.iter().map(|e| e.src.0 as u32).collect(),
        count: 100,
    };
    let decoded = decoder.decode_spikes(&buffer);
    assert!(!decoded.is_empty());
    assert!(decoded.contains(&"hallo".to_string()));
    assert!(decoded.contains(&"hund".to_string()));
    assert!(decoded.contains(&"läuft".to_string()));

    let mut conv = ConversationBuffer::new(10);
    conv.push(ConversationTurn::new_user("hallo".to_string()));
    let json = conv.to_json();
    let imported = ConversationBuffer::from_json(&json, 10).unwrap();
    assert_eq!(imported.len(), 1);
    assert_eq!(imported.turns()[0].text, "hallo");

    transitional.observe(&vec!["hund".to_string(), "läuft".to_string()]);
    assert_eq!(transitional.size(), 1);
    let mut rng = rand::thread_rng();
    let generated = transitional.generate("hund", 4, &mut rng);
    assert_eq!(generated.len(), 4);
    assert_eq!(generated[0], "hund");
}

#[test]
fn conversation_flow_simulation() {
    let mut conv = ConversationBuffer::new(20);

    conv.push(ConversationTurn::new_user("Hallo!".to_string()));
    conv.push(ConversationTurn::new_assistant("Hi there!".to_string(), vec!["hi".to_string()]));
    conv.push(ConversationTurn::new_user("Wie geht's?".to_string()));

    assert_eq!(conv.len(), 3);
    assert_eq!(conv.turns()[0].role, "user");
    assert_eq!(conv.turns()[0].text, "Hallo!");
    assert_eq!(conv.turns()[1].role, "assistant");
    assert_eq!(conv.turns()[1].text, "Hi there!");
    assert_eq!(conv.turns()[2].role, "user");
    assert_eq!(conv.turns()[2].text, "Wie geht's?");

    let json = conv.to_json();
    assert!(json.contains("Hallo!"));
    assert!(json.contains("Hi there!"));
    assert!(json.contains("Wie geht's?"));
}

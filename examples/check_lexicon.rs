use goldworm::{SemanticTrainer, TokenSpikeEncoder, SpikeTokenDecoder, build_extended_lexicon};

fn main() {
    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(1.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
    println!("Lexicon size: {}", trainer.lexicon.tokens.len());
    
    let english_count = trainer.lexicon.tokens.iter().filter(|t| {
        let s = &t.surface;
        !s.contains('ä') && !s.contains('ö') && !s.contains('ü') && !s.contains('ß')
    }).count();
    println!("Likely English words: {}", english_count);
    
    let german_count = trainer.lexicon.tokens.iter().filter(|t| {
        let s = &t.surface;
        s.contains('ä') || s.contains('ö') || s.contains('ü') || s.contains('ß') || s == "und" || s == "oder" || s == "aber" || s == "weil" || s == "denn" || s == "sodass" || s == "ist" || s == "bin" || s == "bist" || s == "sind" || s == "seid" || s == "war" || s == "waren" || s == "läuft" || s == "springt" || s == "schläft" || s == "fliegt" || s == "scheint" || s == "wächst" || s == "geht" || s == "kommt" || s == "sieht" || s == "sagt" || s == "hund" || s == "katze" || s == "vogel" || s == "stern" || s == "baum" || s == "blume" || s == "wasser" || s == "feuer" || s == "erde" || s == "haus" || s == "auto" || s == "buch" || s == "tisch" || s == "stuhl" || s == "fenster" || s == "tür" || s == "licht" || s == "luft" || s == "sonne" || s == "mond" || s == "wolke" || s == "regen" || s == "schnee" || s == "wind" || s == "meer" || s == "berg" || s == "wald" || s == "feld" || s == "stadt" || s == "dorf" || s == "straße" || s == "brücke" || s == "zug" || s == "flugzeug" || s == "schiff" || s == "gut" || s == "schlecht" || s == "groß" || s == "klein" || s == "schnell" || s == "langsam" || s == "warm" || s == "kalt" || s == "hell" || s == "dunkel" || s == "wer" || s == "was" || s == "wo" || s == "wann" || s == "warum" || s == "wie" || s == "wen" || s == "wem" || s == "wessen" || s == "welcher" || s == "hallo" || s == "hi" || s == "guten_tag" || s == "moin" || s == "servus" || s == "ich" || s == "du" || s == "er" || s == "sie" || s == "es" || s == "wir" || s == "ihr" || s == "mich"
    }).count();
    println!("Likely German words: {}", german_count);
}

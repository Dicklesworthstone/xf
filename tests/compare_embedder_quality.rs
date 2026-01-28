//! Compare semantic quality across different embedders

use xf::embedder::Embedder;
use xf::hash_embedder::HashEmbedder;
use xf::model_registry::ModelRegistry;
use xf::model2vec_embedder::Model2VecEmbedder;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[allow(clippy::cast_precision_loss)]
fn test_embedder(name: &str, embedder: &dyn Embedder) {
    println!("\n============================================================");
    println!("Testing: {} ({}d)", name, embedder.dimension());
    println!("============================================================");

    let test_pairs = vec![
        // Similar (should score HIGH)
        (
            "I love programming in Rust",
            "Rust is my favorite language",
            true,
        ),
        ("The cat sat on the mat", "A feline rested on the rug", true),
        (
            "Machine learning is great",
            "AI and deep learning rock",
            true,
        ),
        // Dissimilar (should score LOW)
        ("I love programming", "The cat sat on mat", false),
        ("Stock market crashed", "Beautiful sunny day", false),
        ("Quantum physics", "Buy groceries today", false),
    ];

    let mut similar_scores = Vec::new();
    let mut dissimilar_scores = Vec::new();

    for (a, b, should_be_similar) in &test_pairs {
        let emb_a = embedder.embed(a).unwrap();
        let emb_b = embedder.embed(b).unwrap();
        let sim = cosine_similarity(&emb_a, &emb_b);

        if *should_be_similar {
            similar_scores.push(sim);
        } else {
            dissimilar_scores.push(sim);
        }
    }

    let avg_similar: f32 = similar_scores.iter().sum::<f32>() / similar_scores.len() as f32;
    let avg_dissimilar: f32 =
        dissimilar_scores.iter().sum::<f32>() / dissimilar_scores.len() as f32;
    let separation = avg_similar - avg_dissimilar;

    println!("Similar pairs avg:     {avg_similar:.3}");
    println!("Dissimilar pairs avg:  {avg_dissimilar:.3}");
    println!("Separation gap:        {separation:.3}");
    println!(
        "Quality: {}",
        if separation > 0.2 {
            "GOOD"
        } else if separation > 0.1 {
            "OK"
        } else {
            "POOR"
        }
    );
}

fn test_ranking(name: &str, embedder: &dyn Embedder) {
    let query = "How do I debug memory leaks?";
    let docs = [
        ("Memory profiling with Valgrind", true),
        ("Finding memory leaks in C++", true),
        ("Chocolate cake recipe", false),
        ("Ancient Roman history", false),
    ];

    println!("\n{name}");
    let q_emb = embedder.embed(query).unwrap();

    let mut scored: Vec<_> = docs
        .iter()
        .map(|(doc, rel)| {
            let d_emb = embedder.embed(doc).unwrap();
            (*doc, *rel, cosine_similarity(&q_emb, &d_emb))
        })
        .collect();

    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (doc, rel, score) in &scored {
        let marker = if *rel { "REL" } else { "IRR" };
        println!("  {score:.3} [{marker}] {doc}");
    }

    let rel_min = scored
        .iter()
        .filter(|(_, r, _)| *r)
        .map(|(_, _, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let irr_max = scored
        .iter()
        .filter(|(_, r, _)| !*r)
        .map(|(_, _, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);

    if rel_min > irr_max {
        println!("  => Perfect ranking!");
    } else {
        println!("  => RANKING FAILED");
    }
}

#[test]
#[ignore = "requires model files"]
fn compare_all_embedders() {
    println!("\n\n========== EMBEDDER QUALITY COMPARISON ==========\n");

    // Hash embedder (baseline - no semantics)
    let hash = HashEmbedder::default();
    test_embedder("hash-fnv1a-384 (NO SEMANTICS)", &hash);

    // Model2Vec
    if let Ok(m2v) = Model2VecEmbedder::try_load("potion-retrieval-32M") {
        test_embedder("potion-retrieval-32M (Model2Vec)", &m2v);
    }

    if let Ok(m2v) = Model2VecEmbedder::try_load("potion-multilingual-128M") {
        test_embedder("potion-multilingual-128M (Model2Vec)", &m2v);
    }

    // Transformers via registry
    let registry = ModelRegistry::new();

    for model_name in ["all-MiniLM-L6-v2", "bge-small-en-v1.5"] {
        let config = xf::model_registry::EmbedderConfig::new(model_name);
        if let Ok(embedder) = registry.embedder(&config) {
            test_embedder(&format!("{model_name} (Transformer)"), embedder.as_ref());
        }
    }

    println!("\n\n========== RANKING TEST ==========");
    println!("Query: \"How do I debug memory leaks?\"\n");

    // Hash
    test_ranking("hash-fnv1a-384", &hash);

    // Model2Vec
    if let Ok(m2v) = Model2VecEmbedder::try_load("potion-retrieval-32M") {
        test_ranking("potion-retrieval-32M", &m2v);
    }

    // Transformer
    let config = xf::model_registry::EmbedderConfig::new("all-MiniLM-L6-v2");
    if let Ok(embedder) = registry.embedder(&config) {
        test_ranking("all-MiniLM-L6-v2", embedder.as_ref());
    }
}

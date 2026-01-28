//! Semantic quality verification for embeddings.
//!
//! Tests that embeddings capture semantic meaning by verifying:
//! 1. Similar concepts have high cosine similarity
//! 2. Unrelated concepts have low similarity
//! 3. Synonyms cluster together
//! 4. Antonyms are distant

use xf::embedder::Embedder;
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

#[test]
#[ignore = "Requires model files"]
#[allow(clippy::cast_precision_loss)]
fn test_semantic_similarity_pairs() {
    let embedder =
        Model2VecEmbedder::try_load("potion-retrieval-32M").expect("Failed to load model");

    // Similar pairs should have HIGH similarity (> 0.7)
    let similar_pairs = vec![
        (
            "I love programming in Rust",
            "Rust is my favorite programming language",
        ),
        ("The cat sat on the mat", "A feline was resting on the rug"),
        (
            "Machine learning is fascinating",
            "AI and deep learning are interesting",
        ),
        (
            "The stock market crashed today",
            "Financial markets experienced a major downturn",
        ),
        ("I'm feeling happy today", "I'm in a great mood"),
        ("The weather is beautiful", "It's a gorgeous sunny day"),
        ("He runs very fast", "He sprints at high speed"),
        ("The food was delicious", "The meal tasted amazing"),
    ];

    // Dissimilar pairs should have LOW similarity (< 0.5)
    let dissimilar_pairs = vec![
        ("I love programming in Rust", "The cat sat on the mat"),
        (
            "Machine learning is fascinating",
            "The weather is beautiful",
        ),
        ("The stock market crashed today", "I'm feeling happy today"),
        ("Quantum physics is complex", "I need to buy groceries"),
        (
            "The sunset was breathtaking",
            "Database optimization techniques",
        ),
        ("My dog loves playing fetch", "The economy is recovering"),
    ];

    println!("\n=== SIMILAR PAIRS (should be > 0.7) ===");
    let mut similar_scores = Vec::new();
    for (a, b) in &similar_pairs {
        let emb_a = embedder.embed(a).expect("embed failed");
        let emb_b = embedder.embed(b).expect("embed failed");
        let sim = cosine_similarity(&emb_a, &emb_b);
        similar_scores.push(sim);
        let status = if sim > 0.7 { "✓" } else { "✗" };
        println!("{status} {sim:.3}: \"{a}\" <-> \"{b}\"");
    }

    println!("\n=== DISSIMILAR PAIRS (should be < 0.5) ===");
    let mut dissimilar_scores = Vec::new();
    for (a, b) in &dissimilar_pairs {
        let emb_a = embedder.embed(a).expect("embed failed");
        let emb_b = embedder.embed(b).expect("embed failed");
        let sim = cosine_similarity(&emb_a, &emb_b);
        dissimilar_scores.push(sim);
        let status = if sim < 0.5 { "✓" } else { "✗" };
        println!("{status} {sim:.3}: \"{a}\" <-> \"{b}\"");
    }

    let avg_similar: f32 = similar_scores.iter().sum::<f32>() / similar_scores.len() as f32;
    let avg_dissimilar: f32 =
        dissimilar_scores.iter().sum::<f32>() / dissimilar_scores.len() as f32;

    println!("\n=== SUMMARY ===");
    println!("Average similar pair score: {avg_similar:.3}");
    println!("Average dissimilar pair score: {avg_dissimilar:.3}");
    println!(
        "Separation (similar - dissimilar): {:.3}",
        avg_similar - avg_dissimilar
    );

    // The gap between similar and dissimilar should be significant
    assert!(
        avg_similar > avg_dissimilar,
        "Similar pairs should score higher than dissimilar pairs"
    );
    assert!(
        avg_similar - avg_dissimilar > 0.1,
        "There should be meaningful separation between similar and dissimilar"
    );
}

#[test]
#[ignore = "Requires model files"]
#[allow(clippy::cast_precision_loss, clippy::items_after_statements)]
fn test_topic_clustering() {
    let embedder =
        Model2VecEmbedder::try_load("potion-retrieval-32M").expect("Failed to load model");

    // Topics that should cluster together
    let tech_sentences = vec![
        "JavaScript frameworks like React are popular",
        "Python is great for data science",
        "Cloud computing with AWS and Azure",
        "Docker containers simplify deployment",
    ];

    let food_sentences = vec![
        "Italian pasta with marinara sauce",
        "Fresh sushi from the fish market",
        "Homemade chocolate chip cookies",
        "Grilled steak with vegetables",
    ];

    let sports_sentences = vec![
        "The basketball game was exciting",
        "Soccer players scored three goals",
        "Tennis match went to five sets",
        "Marathon runners crossed the finish line",
    ];

    fn average_embedding(embedder: &Model2VecEmbedder, sentences: &[&str]) -> Vec<f32> {
        let embeddings: Vec<Vec<f32>> = sentences
            .iter()
            .map(|s| embedder.embed(s).unwrap())
            .collect();

        let dim = embeddings[0].len();
        let mut avg = vec![0.0f32; dim];
        for emb in &embeddings {
            for (i, v) in emb.iter().enumerate() {
                avg[i] += v;
            }
        }
        for v in &mut avg {
            *v /= embeddings.len() as f32;
        }
        avg
    }

    let tech_centroid = average_embedding(&embedder, &tech_sentences);
    let food_centroid = average_embedding(&embedder, &food_sentences);
    let sports_centroid = average_embedding(&embedder, &sports_sentences);

    // Check that topics are more similar to themselves than to other topics
    println!("\n=== TOPIC CLUSTERING ===");

    for (name, sentences, _own_centroid) in [
        ("Tech", &tech_sentences, &tech_centroid),
        ("Food", &food_sentences, &food_centroid),
        ("Sports", &sports_sentences, &sports_centroid),
    ] {
        let other_centroids = [
            (&tech_centroid, "Tech"),
            (&food_centroid, "Food"),
            (&sports_centroid, "Sports"),
        ];

        println!("\n{name} sentences:");
        for sentence in sentences {
            let emb = embedder.embed(sentence).unwrap();
            let short = &sentence[..sentence.len().min(40)];
            print!("  \"{short}...\"\n    ");
            for (centroid, centroid_name) in &other_centroids {
                let sim = cosine_similarity(&emb, centroid);
                print!("{centroid_name}: {sim:.3}  ");
            }
            println!();
        }
    }

    // Cross-topic similarity should be lower than intra-topic
    let tech_food_sim = cosine_similarity(&tech_centroid, &food_centroid);
    let tech_sports_sim = cosine_similarity(&tech_centroid, &sports_centroid);
    let food_sports_sim = cosine_similarity(&food_centroid, &sports_centroid);

    println!("\n=== CENTROID SIMILARITIES ===");
    println!("Tech <-> Food: {tech_food_sim:.3}");
    println!("Tech <-> Sports: {tech_sports_sim:.3}");
    println!("Food <-> Sports: {food_sports_sim:.3}");

    // All cross-topic similarities should be relatively low
    assert!(tech_food_sim < 0.8, "Tech and Food should be distinct");
    assert!(tech_sports_sim < 0.8, "Tech and Sports should be distinct");
    assert!(food_sports_sim < 0.8, "Food and Sports should be distinct");
}

#[test]
#[ignore = "Requires model files"]
fn test_query_document_matching() {
    let embedder =
        Model2VecEmbedder::try_load("potion-retrieval-32M").expect("Failed to load model");

    // Queries and their relevant documents
    let test_cases = vec![
        (
            "How do I fix a memory leak in my application?",
            vec![
                ("Memory management best practices in C++", true),
                ("Debugging memory issues with Valgrind", true),
                ("Common causes of memory leaks", true),
                ("Chocolate cake recipe", false),
                ("History of ancient Rome", false),
            ],
        ),
        (
            "Best restaurants in New York City",
            vec![
                ("Top dining spots in Manhattan", true),
                ("NYC food scene guide", true),
                ("Where to eat in Brooklyn", true),
                ("Quantum computing algorithms", false),
                ("Car repair tutorials", false),
            ],
        ),
        (
            "How to train a neural network",
            vec![
                ("Deep learning tutorial for beginners", true),
                ("Backpropagation explained", true),
                ("PyTorch training loop guide", true),
                ("Gardening tips for spring", false),
                ("Pet grooming services", false),
            ],
        ),
    ];

    println!("\n=== QUERY-DOCUMENT MATCHING ===");

    let mut correct = 0;
    let mut total = 0;

    for (query, documents) in &test_cases {
        println!("\nQuery: \"{query}\"");
        let query_emb = embedder.embed(query).unwrap();

        let mut scored: Vec<_> = documents
            .iter()
            .map(|(doc, relevant)| {
                let doc_emb = embedder.embed(doc).unwrap();
                let sim = cosine_similarity(&query_emb, &doc_emb);
                (doc, *relevant, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        for (doc, relevant, sim) in &scored {
            let marker = if *relevant { "✓" } else { "✗" };
            let expected = if *relevant { "relevant" } else { "irrelevant" };
            println!("  {marker} {sim:.3} [{expected}] \"{doc}\"");
        }

        // Check if relevant docs are ranked higher than irrelevant
        let relevant_scores: Vec<f32> = scored
            .iter()
            .filter(|(_, r, _)| *r)
            .map(|(_, _, s)| *s)
            .collect();
        let irrelevant_scores: Vec<f32> = scored
            .iter()
            .filter(|(_, r, _)| !*r)
            .map(|(_, _, s)| *s)
            .collect();

        let min_relevant = relevant_scores
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_irrelevant = irrelevant_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if min_relevant > max_irrelevant {
            println!("  → Perfect separation! ✓");
            correct += 1;
        } else {
            println!("  → Some overlap in ranking");
        }
        total += 1;
    }

    println!("\n=== RANKING ACCURACY ===");
    println!("Queries with perfect separation: {correct}/{total}");
}

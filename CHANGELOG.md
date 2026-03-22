# Changelog

All notable changes to **xf** are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) | Versioning: [SemVer](https://semver.org/spec/v2.0.0.html)

Repository: <https://github.com/Dicklesworthstone/xf>

---

## [Unreleased]

277 commits since v0.2.0 (2026-01-12 through 2026-03-06). No release tag yet.

Compare: [`v0.2.0...main`](https://github.com/Dicklesworthstone/xf/compare/v0.2.0...main)

### Semantic Search Pipeline (Major)

- **Embedding bake-off infrastructure**: benchmark runner with MAP, coefficient of variation, bootstrap CI, ground truth generation, dev/test corpus splits ([`8034cfb`](https://github.com/Dicklesworthstone/xf/commit/8034cfb57c35b75e7876c8899dfb5ed629344513), [`b7a8901`](https://github.com/Dicklesworthstone/xf/commit/b7a89011f2609510716040d4c5aa3fd7e820d756), [`eaf3845`](https://github.com/Dicklesworthstone/xf/commit/eaf3845f134dc30796a5702c84d3c963c5aa1fd8))
- **Embedding backends**: static-retrieval-mrl-en-v1 ONNX ([`9659ec5`](https://github.com/Dicklesworthstone/xf/commit/9659ec5af755ec30110ae1b526681e87686d6689)), Model2Vec potion-retrieval-32M ([`9f69051`](https://github.com/Dicklesworthstone/xf/commit/9f690519ab7882f6b5b4c8c44780390610517ea5)), FastEmbed ([`94143a0`](https://github.com/Dicklesworthstone/xf/commit/94143a06a43f5ffd85adbb0e02623c5ffad78aaa)), fastembed/model2vec batch processing with error recovery ([`58f7073`](https://github.com/Dicklesworthstone/xf/commit/58f70735c2957972fa097949982be8fdc27ccf9b))
- **Benchmark baselines**: MiniLM ([`c205a37`](https://github.com/Dicklesworthstone/xf/commit/c205a37f314e7562f05fc66bfb990ef50b3500c3)), bge-small-en-v1.5 ([`22df100`](https://github.com/Dicklesworthstone/xf/commit/22df100970af0440a77f23cad9ad5a7b881060c5)), nomic-embed-text-v1.5 ([`700cd42`](https://github.com/Dicklesworthstone/xf/commit/700cd422c11fa6a452cc6b5e4dedf539c9bbce07)), EmbeddingGemma-300M ([`681a3a7`](https://github.com/Dicklesworthstone/xf/commit/681a3a71acab808c6d99278eda567e32b4deee3b)), Model2Vec ([`841a530`](https://github.com/Dicklesworthstone/xf/commit/841a5306a30f860d5bd3e0fe3b012a7c27f57239)), transformer embedders ([`ea16127`](https://github.com/Dicklesworthstone/xf/commit/ea161275f37c7db12b1c8edb24a33967e81bcf4f))
- **Bake-off report**: comprehensive winner selection published ([`cbbb146`](https://github.com/Dicklesworthstone/xf/commit/cbbb1463f46e5fc8d2f5f328cf667357bf029199))
- **Vector similarity module**: frankensearch-embed integration ([`bd87cb7`](https://github.com/Dicklesworthstone/xf/commit/bd87cb7805e7558df12d29a92508faec452e7333))
- **Model registry**: ModelCategory enum, convenience methods ([`a15b71c`](https://github.com/Dicklesworthstone/xf/commit/a15b71c37ba02da0fa309870b3bf03a410cc8072), [`48d5342`](https://github.com/Dicklesworthstone/xf/commit/48d5342c74c4bece9165dd3ce18cfc8ed8ed18b5))

### Reranker

- **FlashRank nano cross-encoder** (bd-23l) ([`5ffa662`](https://github.com/Dicklesworthstone/xf/commit/5ffa6625e8e2f35993a5e5c761858d8b984933ec))
- **mxbai-rerank-xsmall-v1 ONNX** backend ([`e279cf4`](https://github.com/Dicklesworthstone/xf/commit/e279cf44f15d4acbd26a4f8afd02afaa7b100b6c))
- **RerankStep** for search pipeline integration ([`0f1ad91`](https://github.com/Dicklesworthstone/xf/commit/0f1ad915791a99d40f38cda31e5a13c5cc9b4876))
- **Quality evaluation framework** for rerankers ([`398ffb4`](https://github.com/Dicklesworthstone/xf/commit/398ffb4d0b436a25dc5383bcab3bd0cc9d1916c9))

### Daemon Mode

- **Daemon core**: Unix domain socket (UDS) listener and protocol ([`fa2dea5`](https://github.com/Dicklesworthstone/xf/commit/fa2dea5e34e68bf212f017c02c4ea2cba14dc48c))
- **CLI commands** for daemon management: `xf daemon start|stop|status` ([`5c315ab`](https://github.com/Dicklesworthstone/xf/commit/5c315ab356cefd029a5ef6e1bfa3ed3ab3b717a0))
- **Client with auto-spawn** and retry logic ([`6c2bedd`](https://github.com/Dicklesworthstone/xf/commit/6c2bedd97d44357ce2d26f8d184d8b620e80abfd))
- **Resource management**: nice/ionice/threads controls ([`d7e98fa`](https://github.com/Dicklesworthstone/xf/commit/d7e98fa3b73c5c601ee03467ec73e3725fa6e859))
- **systemd/launchd service files** ([`8c52765`](https://github.com/Dicklesworthstone/xf/commit/8c52765573be913290b5313c89d85e4ffd433835))
- **Background embedding worker** with job support via Unix socket ([`31b1556`](https://github.com/Dicklesworthstone/xf/commit/31b1556eba0a5eb595ea8d32cc7417c2f6319ca7), [`788f602`](https://github.com/Dicklesworthstone/xf/commit/788f602d87ee7e12e8cc8107e694b5bd300b7d38))
- **TOCTOU race fix** in in-flight request limiting ([`2216518`](https://github.com/Dicklesworthstone/xf/commit/2216518ea9492b1874027fbe5a1969841a42a3e6))
- **Atomic counters extraction** from DaemonState for sync Drop ([`caf5b33`](https://github.com/Dicklesworthstone/xf/commit/caf5b3356581cbaeef70231b6564a68f370daebb))
- Comprehensive test suites: stress tests, LRU eviction, memory pressure, client config, reranker/MRL dimension E2E, 23 unit tests for coverage audit, semantic pipeline tests ([`78903c3`](https://github.com/Dicklesworthstone/xf/commit/78903c377e066e54cd0c1a53ed5cbff8cdde2f67), [`015e088`](https://github.com/Dicklesworthstone/xf/commit/015e0880028e28a6afd25bc59f3d478427f0f0a6), [`439c78f`](https://github.com/Dicklesworthstone/xf/commit/439c78fce19a29ecc29b9db6bc50a11efbf102af), [`f008141`](https://github.com/Dicklesworthstone/xf/commit/f00814135d94cf844035914e4d73cf98ed98244d))

### Two-Tier Progressive Search

- Fast+quality model blending with RRF fusion ([`2789c39`](https://github.com/Dicklesworthstone/xf/commit/2789c39d895ede37927b1aeae6ddf9b289848ec0))
- CLI wiring and convenience flags ([`c93a1e5`](https://github.com/Dicklesworthstone/xf/commit/c93a1e516a882a92bdd99b790c6fb3ce8cee47e9), [`23695ca`](https://github.com/Dicklesworthstone/xf/commit/23695ca1337156fec97c0f27aef1fcd1e99e88fc))
- TwoTierMetrics for search analytics and tracking ([`4b28bad`](https://github.com/Dicklesworthstone/xf/commit/4b28bad0527b5dc341c2cd3e6a5e15714efb021a))

### Rich Terminal Output (rich_rust integration)

- **Output abstraction module** with theme system ([`aae92d1`](https://github.com/Dicklesworthstone/xf/commit/aae92d1f62c1567246c5319862b5617fbbb91746), [`701b921`](https://github.com/Dicklesworthstone/xf/commit/701b9218c688ea80bcdf68ecae493a12e52d32e6))
- **toon_rust integration** for enhanced terminal output ([`9790533`](https://github.com/Dicklesworthstone/xf/commit/9790533e4a316e88bd15942b5b526a93a8e3cfd2))
- Styled components: search result cards ([`e8da940`](https://github.com/Dicklesworthstone/xf/commit/e8da940cb7d40f397a0c31444fa18216eb428f83)), tweet cards ([`af56fd3`](https://github.com/Dicklesworthstone/xf/commit/af56fd3f2d23b566528bc9ae756d0575c37f56af)), stats dashboard ([`c02a906`](https://github.com/Dicklesworthstone/xf/commit/c02a906705a0e18abd15330210a0fc8d769e29b0)), error/warning panels ([`f232fbc`](https://github.com/Dicklesworthstone/xf/commit/f232fbc1a0f2bdca6eaf6303c5b2462f33d6dddf)), progress bars/spinners ([`8bbbb70`](https://github.com/Dicklesworthstone/xf/commit/8bbbb700b45f34ec520d53386e681892472e0754)), doctor renderer ([`ca26136`](https://github.com/Dicklesworthstone/xf/commit/ca26136f8c4693578ad43b1d05dd8d287dd347d4))
- Multi-level verbosity and quiet-mode progress suppression ([`cf780a0`](https://github.com/Dicklesworthstone/xf/commit/cf780a08b6da9c8951aeaa850d9bd2c512f836ad))
- Full Output abstraction migration across all CLI commands ([`6b6a68a`](https://github.com/Dicklesworthstone/xf/commit/6b6a68a8b7a4ad6b11069861846b55bb782953d2), [`b5082ba`](https://github.com/Dicklesworthstone/xf/commit/b5082bac69ed8d53bf141850c82aa1c153157af3))

### CLI Enhancements

- **`xf models list` / `xf models info`** subcommands for semantic search config ([`5ff08c6`](https://github.com/Dicklesworthstone/xf/commit/5ff08c6f8dd7cb5a7ec39788f69398eb87b5f47f), [`b68fa1c`](https://github.com/Dicklesworthstone/xf/commit/b68fa1c0890a1f280b472e45444d4797fa801a05))
- **Shell completions** generation (bash, zsh, fish, PowerShell) ([`a9ad4bc`](https://github.com/Dicklesworthstone/xf/commit/a9ad4bccf9cbd4137286b4a27e7a3c6846238272))
- **`xf robot-docs`** for machine-readable documentation ([`3a14ef5`](https://github.com/Dicklesworthstone/xf/commit/3a14ef56a5d9aadbb3765b55eab68107fd053016))
- REPL enhancements: search modes, index status, styled output ([`9576872`](https://github.com/Dicklesworthstone/xf/commit/9576872f047ecb2a2cc9b391084c45e594ec29ec))
- Environment variable documentation in `--help` text ([`673672c`](https://github.com/Dicklesworthstone/xf/commit/673672c93bb6cf2c42348669a5407643c3262c2d))

### Vector Index

- Persistent vector index writer/reader with file loading and mmap support ([`d7ec94a`](https://github.com/Dicklesworthstone/xf/commit/d7ec94ae089fa5be862b8a8950786948d9149981), [`249f4b8`](https://github.com/Dicklesworthstone/xf/commit/249f4b8edc07f4a4bea40d3be21ae75ef9456aaa))
- Wired into search path with doctor checks ([`f56793d`](https://github.com/Dicklesworthstone/xf/commit/f56793d00aa8c285e77dcacb578e48ec668ee590))
- Type-aware RRF fusion and type counts ([`c8bb438`](https://github.com/Dicklesworthstone/xf/commit/c8bb4384dd9a1fce42d412664e9bca98143a8f92), [`e346c0d`](https://github.com/Dicklesworthstone/xf/commit/e346c0d7acb3e229b327ac529d19bc877ee1b98b))
- Cache invalidation, stale detection, and init logging ([`9a66bbd`](https://github.com/Dicklesworthstone/xf/commit/9a66bbd5b9bb7fb1c79f8864b2b63773ff5d2752), [`6373fbd`](https://github.com/Dicklesworthstone/xf/commit/6373fbdccb1e765f511ea94bd1bf23b3fc5a1f32))
- Batch embedding hash lookup and preloading ([`cfaca28`](https://github.com/Dicklesworthstone/xf/commit/cfaca2895c139dc69e5f2612bcbe566e456075a7), [`bdf2d43`](https://github.com/Dicklesworthstone/xf/commit/bdf2d43c8bb386b8d879356ab66ea2930876b8fe))

### Performance

- Batch `get_by_ids` for semantic/hybrid lookups ([`e1802d7`](https://github.com/Dicklesworthstone/xf/commit/e1802d7e13e3ede6f069c55743aa98b7e10ba930), [`faf73f8`](https://github.com/Dicklesworthstone/xf/commit/faf73f8113deb463d142ff22f5b5bed3726bda48))
- RRF alloc/perf instrumentation and isomorphism coverage ([`37efca1`](https://github.com/Dicklesworthstone/xf/commit/37efca102d01769268c699ccfb1d40d2a8000ef7), [`56dea4c`](https://github.com/Dicklesworthstone/xf/commit/56dea4c30cc8d45b4888e9e4fd35a734cd0949f1))
- Consolidated stats counts in storage ([`baa3a50`](https://github.com/Dicklesworthstone/xf/commit/baa3a50cf343c8e637a97be821c45b90c42f0077))
- Type-filtered search benchmarks ([`51394d8`](https://github.com/Dicklesworthstone/xf/commit/51394d8c825301a409361c502bc6fab122be76f8))
- Performance baseline report and corpus benchmarks ([`3d5fb5d`](https://github.com/Dicklesworthstone/xf/commit/3d5fb5ddcf94405d87007d6b0e0e4749f8c864b8), [`0895cd2`](https://github.com/Dicklesworthstone/xf/commit/0895cd25a6466af52c1b620c83f3913c7e9e82e1))

### Bug Fixes

- Scope embedding hash lookups by model_id to prevent cross-model dedup collisions ([`4ea7ac6`](https://github.com/Dicklesworthstone/xf/commit/4ea7ac689fc3e16cfe33651ad94838843b7f563b))
- Don't strip markup from data output to preserve JSON brackets ([`bc66900`](https://github.com/Dicklesworthstone/xf/commit/bc669003782938a6f7bfe26d58f5ebc1578d1484))
- Schema migration order + embedding rate in progress bar ([`6c70ea0`](https://github.com/Dicklesworthstone/xf/commit/6c70ea0a5ad0e3802acdc613adbeaf72ed34e674))
- Populate type_counts to fix type-filtered semantic search ([`6fc0f77`](https://github.com/Dicklesworthstone/xf/commit/6fc0f779fa6b33cbc0f034ee71614b81667a1d37))
- Make search deterministic for same-id different-type documents ([`e8a6386`](https://github.com/Dicklesworthstone/xf/commit/e8a6386fa6f24d428e34131ad32836d5b182c760))
- Route tracing logs to stderr instead of stdout ([`3874662`](https://github.com/Dicklesworthstone/xf/commit/3874662ce1093e7215e6c3b6ebbd0e23f0a7f148))
- Respect `search.highlight` config ([`e9f1d68`](https://github.com/Dicklesworthstone/xf/commit/e9f1d68b2f6d24bffb7ee21e6c85699a3ad1b9e8))
- Exclude likes from date-filtered results ([`738dc85`](https://github.com/Dicklesworthstone/xf/commit/738dc8506edd4c97bce4b2e0dbbec5bdce713050))
- Keep JSON output for empty search results ([`92f43b0`](https://github.com/Dicklesworthstone/xf/commit/92f43b0ca289d0e171215ea7964c8f7b90afabeb))
- Treat date-only natural language as day range ([`a0264b7`](https://github.com/Dicklesworthstone/xf/commit/a0264b7f05650b7831d550848a47243908b3fbfa))
- Skip empty likes in index/FTS ([`f87df8d`](https://github.com/Dicklesworthstone/xf/commit/f87df8de62b08f5a50261a0ce9f523e6c4466a47))
- Validate embeddings before writing vector index ([`cd12645`](https://github.com/Dicklesworthstone/xf/commit/cd1264550efacd86794842f370317fcb6e6da013))
- Bound embedding reuse cache ([`c5f5cc6`](https://github.com/Dicklesworthstone/xf/commit/c5f5cc65a8fb1c07854d22e8ee3f522247fab846))
- Handle phrase queries correctly in REPL search ([`40f772e`](https://github.com/Dicklesworthstone/xf/commit/40f772ebfa9e5117f166a3ec0d16e5de260c74f0))
- Escape glob special characters in archive paths ([`d70d7a3`](https://github.com/Dicklesworthstone/xf/commit/d70d7a33b5c4cb9cfa72a9ddf247c8a460e7aab2), [`85c4c0b`](https://github.com/Dicklesworthstone/xf/commit/85c4c0b88e536dc30491c7363abc965d9f4b0f98))
- Handle NULL values from SQLite date/time functions ([`03c251b`](https://github.com/Dicklesworthstone/xf/commit/03c251b87764ae35e449033e9cb1c1e899eb3e74))
- Split prefixes on non-alphanumeric chars for punctuation handling ([`78380ff`](https://github.com/Dicklesworthstone/xf/commit/78380ffbff0fc3199154ee1398abd363061d88ee))
- Avoid `get_by_ids` underfetch for untyped lookups ([`0e3bcc4`](https://github.com/Dicklesworthstone/xf/commit/0e3bcc4ebb8bacdbac005f4ae8b214c0ded442b2))
- Fix `stats index_built_at` and expand tilde paths ([`c8b64d3`](https://github.com/Dicklesworthstone/xf/commit/c8b64d320f94cbce69052189966ffdf4fdfa5396))
- Robust doctor JS wrapper parsing ([`1f1b9af`](https://github.com/Dicklesworthstone/xf/commit/1f1b9af418e06de0486a2157d9362c985b1c3d44))
- Fix JSON output logging ([`a85fc0e`](https://github.com/Dicklesworthstone/xf/commit/a85fc0e38be20e0710b95f08163e65a61c41213a))
- Parse empty object semantics in metadata ([`e33aed9`](https://github.com/Dicklesworthstone/xf/commit/e33aed901882ea5624b8c0639b2202f68bda536f))
- Replace local path deps with git deps for portable builds ([`715318e`](https://github.com/Dicklesworthstone/xf/commit/715318ef3ec4955bd278326c05fe2024c67c612f))

### Refactoring

- Remove legacy search stack after frankensearch cutover ([`0335246`](https://github.com/Dicklesworthstone/xf/commit/033524698dfdf9ece329484f2e8234cda05be3c1))
- Simplify hybrid search logic, remove 10 lines ([`69fc50a`](https://github.com/Dicklesworthstone/xf/commit/69fc50a5df3ef6eb159043f214b61a4e9dfc348b))
- Remove redundant init locks, add quality fallback warning ([`2e2c619`](https://github.com/Dicklesworthstone/xf/commit/2e2c619ce7715629973dbd54774096771c9e7835))
- Optimize JSON parsing, N+1 queries, query patterns, allocations ([`bec1cf1`](https://github.com/Dicklesworthstone/xf/commit/bec1cf1f4425c16df20025266173a472bba5dbd6), [`3c4a11f`](https://github.com/Dicklesworthstone/xf/commit/3c4a11fb4c9ff368968fa8cb3cae3bd2dea1bfdf), [`318386f`](https://github.com/Dicklesworthstone/xf/commit/318386f30e3a629acabca040c1fcc327291183ed), [`eeabb29`](https://github.com/Dicklesworthstone/xf/commit/eeabb292363be518b021e2e4cf01212447a11d4f))
- Compute DOW distribution from daily counts ([`332e82f`](https://github.com/Dicklesworthstone/xf/commit/332e82f79bafdc18e01a6775ed71889bcd8c6ed3))

### CI / Infra

- Secret-based job conditionals removed from release workflow ([`b4cadf2`](https://github.com/Dicklesworthstone/xf/commit/b4cadf21930e10ab1af01f9d0e8946ce75d6ddef))
- GitHub Actions best practices improvements ([`47c5308`](https://github.com/Dicklesworthstone/xf/commit/47c53082cbc123ef8610ad6076430dfe053b3e09))
- `repository_dispatch` triggers for homebrew-tap and scoop-bucket ([`6b33a8e`](https://github.com/Dicklesworthstone/xf/commit/6b33a8e4543793a3d1ace5ea21aac032f4bc4092))
- Notify-ACFS workflow for lesson registry sync ([`4d2524a`](https://github.com/Dicklesworthstone/xf/commit/4d2524a5bb5dee9f8fd1f4a058fea9cfa0978ad8))
- Homebrew/Scoop installation docs in README ([`3cd80ec`](https://github.com/Dicklesworthstone/xf/commit/3cd80ec5e06b9b7824f2fd9f894f5b52184a0c90))

### Dependencies

- Migrate asupersync to canonical repo + crates.io v0.2.0 ([`c7e9872`](https://github.com/Dicklesworthstone/xf/commit/c7e98726e8aad679608a5a84c66c41e9145f111e), [`d14d5e4`](https://github.com/Dicklesworthstone/xf/commit/d14d5e40a17ff1772ad2f0e17091952b0ac95727))
- Update rich_rust from pre-release/git ref to crates.io v0.2.0 ([`38d938a`](https://github.com/Dicklesworthstone/xf/commit/38d938a4af9b8de93e319929b4d6f60cd4cc305b))

### Licensing

- License updated to MIT with OpenAI/Anthropic Rider ([`4f96641`](https://github.com/Dicklesworthstone/xf/commit/4f96641c6189ba2953c4cc135f1d3e21dec59e24), [`2329e37`](https://github.com/Dicklesworthstone/xf/commit/2329e37f739dbe297d6a549c450c76e840bc65f5))

---

## [0.2.0] - 2026-01-12

**GitHub Release**: [xf v0.2.0](https://github.com/Dicklesworthstone/xf/releases/tag/v0.2.0)
| **Tag**: [`v0.2.0`](https://github.com/Dicklesworthstone/xf/tree/v0.2.0)
| **Tag commit**: [`f295aa6`](https://github.com/Dicklesworthstone/xf/commit/f295aa615099319484cbff3e259c12c4a4dd33ee)
| **Compare**: [`v0.1.0...v0.2.0`](https://github.com/Dicklesworthstone/xf/compare/v0.1.0...v0.2.0)
| **Published**: 2026-01-12T07:32:12Z
| 30 commits

### Added

- **`xf import` command**: one-step archive setup from downloaded X data ZIP -- extracts to `~/my_x_history` (customizable with `-o`), auto-indexes, displays welcome box with archive statistics ([`6935e5d`](https://github.com/Dicklesworthstone/xf/commit/6935e5d7c1acadfaff1431f386eb76be12aed009))
- **Semantic/hybrid search with RRF fusion**: hash-based vocabulary similarity out of the box, reciprocal rank fusion combining keyword and vector results ([`3cfa8bf`](https://github.com/Dicklesworthstone/xf/commit/3cfa8bf266c6901e66c7e919e5999d762169e62d))
- **`get_by_id_and_type`** for document disambiguation across types ([`370c7af`](https://github.com/Dicklesworthstone/xf/commit/370c7afa5cb77d39d5ae96f46d5e30193dd34004))
- Performance baseline measurements ([`2dcd1a3`](https://github.com/Dicklesworthstone/xf/commit/2dcd1a35e1a9eff04004464e41dbae72be1224a4))
- Comprehensive semantic/hybrid search documentation ([`89a7320`](https://github.com/Dicklesworthstone/xf/commit/89a732081216bf084204f1127fa0c519892ab7fe))
- Documentation: X archive data limitations section, design philosophy, recipes ([`be23d62`](https://github.com/Dicklesworthstone/xf/commit/be23d627d70bb5bd6cd576e33409676906112278), [`c58baf0`](https://github.com/Dicklesworthstone/xf/commit/c58baf0f5b8f7e23112e9a23664152d67ae3fb87))

### Fixed

- **Embeddings composite primary key**: `(doc_id, doc_type)` prevents collisions where liked tweets could collide with own tweets (schema v2 -> v3) ([`a7847ff`](https://github.com/Dicklesworthstone/xf/commit/a7847ff6bf039435bf6c20dcce6405fc8a58e674))
- Proper migration for embeddings schema v2 ([`08d3250`](https://github.com/Dicklesworthstone/xf/commit/08d3250b2f5694abbf0661a1c62061080d388bfc))
- Match Grok `doc_id` format between embeddings and Tantivy ([`77d90de`](https://github.com/Dicklesworthstone/xf/commit/77d90de7e4c54a60c74ac4f92c3bc7708aacbee3))
- Prevent double-offset in hybrid search mode ([`9324f10`](https://github.com/Dicklesworthstone/xf/commit/9324f10497deb1e865bde797b625ed58cfa69182))
- Empty query search and likes indexing edge cases ([`792e6e1`](https://github.com/Dicklesworthstone/xf/commit/792e6e11ae5c333c8cd9e262ff71164651309bf0))
- Welcome box padding and example path length ([`fd9bfe5`](https://github.com/Dicklesworthstone/xf/commit/fd9bfe5ba9a2c37e91b58b65e52f82e25628a2bb), [`6125485`](https://github.com/Dicklesworthstone/xf/commit/61254853bc41ae22dd1befbd43b7b909b91b6b81))
- Installer: use `.tar.gz` format and `SHA256SUMS` checksum file ([`7a14aac`](https://github.com/Dicklesworthstone/xf/commit/7a14aac2e45ad5f2016468b6e022da6738b89022))

### Changed

- Indexing performance improvements ([`3b2db29`](https://github.com/Dicklesworthstone/xf/commit/3b2db297e8f5b0ae6f657ed2711537090a0646a7))
- CLI help and searchable types documentation ([`3f48dff`](https://github.com/Dicklesworthstone/xf/commit/3f48dff40a5fc360432428f1bd4912934c44f82e))
- Progress feedback improvements ([`d20bfc9`](https://github.com/Dicklesworthstone/xf/commit/d20bfc93a11b4ca95af8a065b8643a1bab6be005))

### Upgrade Notes

Users upgrading from v0.1.0 will have their embeddings regenerated automatically on first search (schema migration v2 -> v3). This is a one-time operation.

### Release Artifacts

| Platform | Asset |
|----------|-------|
| Linux x86_64 (glibc) | `xf-x86_64-unknown-linux-gnu.tar.gz` |
| Linux x86_64 (musl) | `xf-x86_64-unknown-linux-musl.tar.gz` |
| macOS Intel | `xf-x86_64-apple-darwin.tar.gz` |
| macOS Apple Silicon | `xf-aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `xf-x86_64-pc-windows-msvc.zip` |
| Checksums | `SHA256SUMS` |

---

## [0.1.0] - 2026-01-11

**GitHub Release**: [xf v0.1.0](https://github.com/Dicklesworthstone/xf/releases/tag/v0.1.0)
| **Tag**: [`v0.1.0`](https://github.com/Dicklesworthstone/xf/tree/v0.1.0)
| **Tag commit**: [`0c73830`](https://github.com/Dicklesworthstone/xf/commit/0c738307ff43eef39de581a83a7d8bbb30809563)
| **Published**: 2026-01-11T17:36:21Z
| **Initial commit**: [`87ee1af`](https://github.com/Dicklesworthstone/xf/commit/87ee1aff8fb054cd51dcfa2dd89175af9118e27e)
| 70 commits

### Core Search Engine

- **Tantivy-based full-text search** with BM25 ranking ([`87ee1af`](https://github.com/Dicklesworthstone/xf/commit/87ee1aff8fb054cd51dcfa2dd89175af9118e27e))
- **Query syntax**: phrase queries, boolean operators (AND, OR, NOT), prefix/wildcard matching
- **SQLite storage** with FTS5 fallback for metadata queries
- **Parallel parsing** with rayon (~10,000 documents/second)
- **Search result highlighting** ([`7d5d832`](https://github.com/Dicklesworthstone/xf/commit/7d5d83230fa79e56d6db3c895c1884b829edddce))

### Supported Data Types

- Tweets (your posts)
- Likes (tweets you have liked)
- Direct Messages (DM conversations)
- Grok conversations (AI assistant chats)
- Followers / Following lists
- Blocks and Mutes ([`46c2272`](https://github.com/Dicklesworthstone/xf/commit/46c2272ae8e7725dc64a769f9888ee747d2b04e3))

### CLI Commands

- **`xf index`** -- index an X archive directory
- **`xf search`** -- full-text search with type/date/sort filters ([`583e3f7`](https://github.com/Dicklesworthstone/xf/commit/583e3f7256f75436fd7a1b06053518f64d5ff36b))
- **`xf stats`** -- archive overview with temporal, engagement, and content analytics ([`f8f5772`](https://github.com/Dicklesworthstone/xf/commit/f8f5772e3f25b2049ec4920f21935429f0b5c0a0), [`9693856`](https://github.com/Dicklesworthstone/xf/commit/96938569e2f347383680f684cc5e0083d1c64b8e), [`96b5bd9`](https://github.com/Dicklesworthstone/xf/commit/96b5bd930d4800e22437d6b087313c741b18617b))
- **`xf stats --detailed`** -- all analytics at once ([`6507b15`](https://github.com/Dicklesworthstone/xf/commit/6507b15e7570452cb9acdb988d6012db3cbb6f06))
- **`xf export`** / **`xf list`** -- export and browse indexed data ([`d54dc60`](https://github.com/Dicklesworthstone/xf/commit/d54dc603b016799253daeb5ac0748d366152e4a1))
- **`xf doctor`** -- health check diagnostics for archive, DB, and index ([`4a47d7b`](https://github.com/Dicklesworthstone/xf/commit/4a47d7b09ee7171ac9ebfe7a4c7cab866ffeea32), [`a95f057`](https://github.com/Dicklesworthstone/xf/commit/a95f0572d91feafb97c873fe051bdd6daff2938d))
- **`xf shell`** -- interactive REPL with tab completion, variables, pipes, session state, `set` command ([`c41f7dc`](https://github.com/Dicklesworthstone/xf/commit/c41f7dcd2c7087f7ffc57fc0fce5384586d58bd8), [`49148f6`](https://github.com/Dicklesworthstone/xf/commit/49148f66866f59fa9d2e031290918bebba6a9eb7), [`df1dd10`](https://github.com/Dicklesworthstone/xf/commit/df1dd103ea673e24dd92885de45ff4a8d45c2f6d), [`d6898be`](https://github.com/Dicklesworthstone/xf/commit/d6898be84a62c32dc3c47140d307a2cdd8d905c0))

### Output Formats

- Text (default, colorized)
- JSON / JSON-pretty
- Compact
- CSV
- `--no-color` flag for plain output ([`82cf0e2`](https://github.com/Dicklesworthstone/xf/commit/82cf0e27a8f29c6c936427ae34beec400ed50bb9))

### UX

- DM context: `--context` flag shows full conversation threads around matches ([`bce70c9`](https://github.com/Dicklesworthstone/xf/commit/bce70c960462e2f5c0c6ca89560ea7e9ee4727d8), [`691f4d1`](https://github.com/Dicklesworthstone/xf/commit/691f4d143f5ea9df4b4e85f4a6c0590330efe473))
- Premium search results display ([`c6548b3`](https://github.com/Dicklesworthstone/xf/commit/c6548b34d287d11fb9ec7daaf46eaf34daa1c3c7))
- Human-friendly data formatting: relative dates, CSV escaping ([`3b30db7`](https://github.com/Dicklesworthstone/xf/commit/3b30db71df0bba5769cb14669527a6451b532254))
- Did-you-mean error suggestions ([`1482f0d`](https://github.com/Dicklesworthstone/xf/commit/1482f0d6f0dd05b27804d7710b2f7f16c3619da4))
- Enhanced REPL startup banner with archive stats ([`5263c4d`](https://github.com/Dicklesworthstone/xf/commit/5263c4d3db8ff45f1c3380ccdc3f38def9cb472c))
- Natural language date parsing (`--since "last month"`, `--until "Q1 2024"`)
- Progress indicators for indexing

### Fixed

- UTF-8 panic in `truncate` when string ends with multi-byte character ([`dbf5a5f`](https://github.com/Dicklesworthstone/xf/commit/dbf5a5fa01517b20278df6d1df20a32af190ef54), [`d601fe1`](https://github.com/Dicklesworthstone/xf/commit/d601fe1291df1e0ba55993cd0b5d7aa34bfa5a56))
- Grok message FTS indexing and search ([`50207cd`](https://github.com/Dicklesworthstone/xf/commit/50207cd9e75c6a78e82cb635d2cad9caf56ffaa0))
- REPL session state bugs: `$_` vs `$_name` substitution conflict, digit-starting variable names ([`9c5d947`](https://github.com/Dicklesworthstone/xf/commit/9c5d947fb5ffc5def029d1f0153f0061d375e4ff), [`17daf7b`](https://github.com/Dicklesworthstone/xf/commit/17daf7bde2c0f81ca4cdf812682844055f40b3af), [`e57438c`](https://github.com/Dicklesworthstone/xf/commit/e57438c8755cb7d0360e68f4adb377227d1f3531))
- Handle small `max_len` in truncate functions ([`981aa78`](https://github.com/Dicklesworthstone/xf/commit/981aa78c7dafbf4f410aed4cd4f7fe56a0a64b0e))
- Non-deterministic doctor suggestions ordering ([`36b2cca`](https://github.com/Dicklesworthstone/xf/commit/36b2cca5432c104c576f2dfadb0abb6de01c8bbf))
- Manifest handling and prefix search ([`d8caa71`](https://github.com/Dicklesworthstone/xf/commit/d8caa71698de1cb80edd0e8fca46337ef5257021))
- Critical issues from deep code review ([`ce26c18`](https://github.com/Dicklesworthstone/xf/commit/ce26c18acc36acee3c0bd20c28749689263eeae3))
- Search limits, schema reads, metadata fallbacks ([`370a611`](https://github.com/Dicklesworthstone/xf/commit/370a611f21f53cbc7d3663ae63d70d3c7d388ea7))

### Changed

- Project renamed from `x_find` to `xf` ([`b26e684`](https://github.com/Dicklesworthstone/xf/commit/b26e6849646d82f4e1f3e35691733f8b5f5ce786))
- Branding updated from Twitter to X throughout ([`9fd586e`](https://github.com/Dicklesworthstone/xf/commit/9fd586e3f1fefcc4e3aad8855b99e17036d07112))

### Technical Details

- Rust Edition 2024 (nightly toolchain required)
- Key dependencies: tantivy 0.22, rusqlite 0.32, clap 4.5, chrono 0.4, rayon 1.10

### Release Artifacts

| Platform | Asset |
|----------|-------|
| Linux x86_64 (glibc) | `xf-x86_64-unknown-linux-gnu.tar.gz` |
| Linux x86_64 (musl) | `xf-x86_64-unknown-linux-musl.tar.gz` |
| macOS Intel | `xf-x86_64-apple-darwin.tar.gz` |
| macOS Apple Silicon | `xf-aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `xf-x86_64-pc-windows-msvc.zip` |
| Checksums | `SHA256SUMS`, `SHA256SUMS.txt`, `SHA512SUMS.txt` |

---

<!-- Link references -->
[Unreleased]: https://github.com/Dicklesworthstone/xf/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Dicklesworthstone/xf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dicklesworthstone/xf/releases/tag/v0.1.0

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

const SAMPLE_TWEETS: &str = r#"window.YTD.tweets.part0 = [
    {
        "tweet": {
            "id_str": "1234567890",
            "created_at": "Wed Jan 08 12:00:00 +0000 2025",
            "full_text": "Hello world! This is a test tweet.",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "1",
            "retweet_count": "0",
            "lang": "en",
            "entities": {"hashtags": [], "user_mentions": [], "urls": []}
        }
    }
]"#;

const SAMPLE_MANIFEST: &str = r#"window.YTD.manifest.part0 = {
    "userInfo": {
        "accountId": "999999999",
        "userName": "test_user",
        "displayName": "Test User"
    },
    "archiveInfo": {
        "sizeBytes": "1234",
        "generationDate": "2025-01-01T00:00:00Z",
        "isPartialArchive": false
    }
}"#;

fn create_minimal_archive() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("tempdir");
    let data_dir = temp_dir.path().join("data");
    fs::create_dir_all(&data_dir).expect("create data dir");
    fs::write(data_dir.join("tweets.js"), SAMPLE_TWEETS).expect("write tweets.js");
    fs::write(data_dir.join("manifest.js"), SAMPLE_MANIFEST).expect("write manifest.js");
    let archive_path = temp_dir.path().to_path_buf();
    (temp_dir, archive_path)
}

fn xf_cmd() -> Command {
    cargo_bin_cmd!("xf")
}

#[test]
fn test_output_format_env_via_cli() {
    let (_archive_temp, archive_path) = create_minimal_archive();
    let storage_dir = TempDir::new().expect("storage tempdir");
    let db_path = storage_dir.path().join("xf.db");
    let index_path = storage_dir.path().join("xf_index");

    let mut index_cmd = xf_cmd();
    index_cmd
        .env("XF_DB", &db_path)
        .env("XF_INDEX", &index_path)
        .arg("index")
        .arg(&archive_path);
    index_cmd.assert().success();

    let output_json = xf_cmd()
        .env("XF_DB", &db_path)
        .env("XF_INDEX", &index_path)
        .arg("stats")
        .arg("--format")
        .arg("json")
        .output()
        .expect("run stats json");
    assert!(output_json.status.success());
    let json_stdout = String::from_utf8_lossy(&output_json.stdout);
    assert!(json_stdout.trim_start().starts_with('{'));

    let output_env = xf_cmd()
        .env("XF_DB", &db_path)
        .env("XF_INDEX", &index_path)
        .env("XF_OUTPUT_FORMAT", "toon")
        .arg("stats")
        .output()
        .expect("run stats with env");
    assert!(output_env.status.success());
    let env_stdout = String::from_utf8_lossy(&output_env.stdout);
    assert!(!env_stdout.trim_start().starts_with('{'));
}

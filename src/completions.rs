//! Shell completions and installation instructions.
//!
//! Generates shell completion scripts and provides styled installation
//! instructions. Completions go to stdout (pipeable), instructions to stderr.

use clap::ValueEnum;
use clap_complete::generate;
use std::io;

use crate::cli::Cli;
use crate::output::Output;
use crate::theme::Theme;

/// Supported shells for completion generation.
///
/// Wraps `clap_complete::Shell` with additional metadata like config paths
/// and installation instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Shell {
    /// Bourne Again SHell
    Bash,
    /// Z Shell
    Zsh,
    /// Friendly Interactive SHell
    Fish,
    /// PowerShell
    #[value(name = "powershell")]
    PowerShell,
    /// Elvish shell
    Elvish,
}

impl Shell {
    /// Get the shell name as a lowercase string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
            Self::PowerShell => "powershell",
            Self::Elvish => "elvish",
        }
    }

    /// Get the typical config file path for this shell.
    #[must_use]
    pub const fn config_file(&self) -> &'static str {
        match self {
            Self::Bash => "~/.bashrc",
            Self::Zsh => "~/.zshrc",
            Self::Fish => "~/.config/fish/config.fish",
            Self::PowerShell => "$PROFILE",
            Self::Elvish => "~/.elvish/rc.elv",
        }
    }

    /// Get installation instructions for this shell.
    #[must_use]
    pub const fn install_instructions(&self) -> &'static str {
        match self {
            Self::Bash => {
                r"Add to ~/.bashrc:
  source <(xf completions bash)

Or save to a file:
  xf completions bash > ~/.local/share/bash-completion/completions/xf"
            }

            Self::Zsh => {
                r"Add to ~/.zshrc:
  source <(xf completions zsh)

Or for oh-my-zsh:
  xf completions zsh > ~/.oh-my-zsh/completions/_xf

Or for general zsh:
  xf completions zsh > ~/.zfunc/_xf
  # Ensure ~/.zfunc is in your fpath"
            }

            Self::Fish => {
                r"Run once:
  xf completions fish | source

Or persist:
  xf completions fish > ~/.config/fish/completions/xf.fish"
            }

            Self::PowerShell => {
                r"Add to $PROFILE:
  xf completions powershell | Out-String | Invoke-Expression

Or save to a file:
  xf completions powershell > xf.ps1
  # Then source it in your profile"
            }

            Self::Elvish => {
                r"Add to ~/.elvish/rc.elv:
  eval (xf completions elvish | slurp)"
            }
        }
    }

    /// Convert to `clap_complete::Shell` for generation.
    #[must_use]
    pub const fn to_clap_shell(&self) -> clap_complete::Shell {
        match self {
            Self::Bash => clap_complete::Shell::Bash,
            Self::Zsh => clap_complete::Shell::Zsh,
            Self::Fish => clap_complete::Shell::Fish,
            Self::PowerShell => clap_complete::Shell::PowerShell,
            Self::Elvish => clap_complete::Shell::Elvish,
        }
    }
}

/// Generate shell completions and show installation instructions.
///
/// Completions are written to stdout (so they can be piped).
/// Installation instructions are written to stderr (so they don't interfere).
pub fn generate_completions(shell: Shell, output: &Output) {
    let mut cmd = <Cli as clap::CommandFactory>::command();

    // Generate completions to stdout
    generate(shell.to_clap_shell(), &mut cmd, "xf", &mut io::stdout());

    // Show installation instructions on stderr
    show_install_instructions(shell, output);
}

/// Display installation instructions to stderr.
fn show_install_instructions(shell: Shell, output: &Output) {
    let instructions = shell.install_instructions();
    let shell_name = shell.as_str();
    let config_file = shell.config_file();

    eprintln!(); // Blank line before instructions

    if output.is_stderr_styled() {
        let theme = Theme::for_terminal();

        // Styled header
        eprintln!(
            "{}",
            theme.format_heading(&format!("Installation Instructions for {shell_name}"))
        );
        eprintln!();

        // Config file hint
        eprintln!(
            "{}",
            theme.format_muted(&format!("Config file: {config_file}"))
        );
        eprintln!();

        // Instructions with some highlighting
        for line in instructions.lines() {
            if line.starts_with("  ") && line.contains("xf completions") {
                // Command lines get primary color
                eprintln!("{}", theme.format_primary(line));
            } else if line.ends_with(':') {
                // Section headers get info color
                eprintln!("{}", theme.format_info(line));
            } else if line.starts_with("  #") || line.starts_with("  //") {
                // Comments get muted color
                eprintln!("{}", theme.format_muted(line));
            } else {
                eprintln!("{line}");
            }
        }
    } else {
        // Plain text output
        eprintln!("# Installation Instructions for {shell_name}");
        eprintln!("# Config file: {config_file}");
        eprintln!();
        eprintln!("{instructions}");
    }
}

/// Render styled help text to the output.
///
/// Provides a more visually appealing help display when running in a TTY.
pub fn render_styled_help(output: &Output) {
    if !output.is_styled() {
        // Fall back to clap's built-in help
        let mut cmd = <Cli as clap::CommandFactory>::command();
        let _ = cmd.print_help();
        return;
    }

    let theme = Theme::for_terminal();

    // Title
    output.print(&format!(
        "{} - {}",
        theme.format_primary("xf"),
        theme.format_heading("X Archive Search Tool")
    ));
    output.print("");

    // Quick summary
    output.print(&theme.format_muted(
        "Ultra-fast CLI for searching your X data archive with sub-millisecond latency.",
    ));
    output.print("");

    // Usage
    output.print(&theme.format_heading("USAGE:"));
    output.print(&format!(
        "    {} [OPTIONS] <COMMAND>",
        theme.format_primary("xf")
    ));
    output.print("");

    // Commands section
    output.print(&theme.format_heading("COMMANDS:"));
    let commands = [
        (
            "search <QUERY>",
            "Search your X archive (tweets, likes, DMs, Grok)",
        ),
        ("stats", "Show archive statistics"),
        ("index <PATH>", "Build or rebuild search index"),
        ("doctor", "Check archive, database, and index health"),
        ("shell", "Launch interactive REPL mode"),
        ("tweet <ID>", "Show details for a specific tweet"),
        ("list <WHAT>", "List available data (tweets, dms, etc.)"),
        ("export <WHAT>", "Export data in various formats"),
        ("completions <SHELL>", "Generate shell completions"),
        ("benchmark", "Run embedding/reranker benchmarks"),
    ];

    for (cmd, desc) in &commands {
        output.print(&format!(
            "    {:24} {}",
            theme.format_primary(cmd),
            theme.format_muted(desc)
        ));
    }
    output.print("");

    // Global options
    output.print(&theme.format_heading("OPTIONS:"));
    let options = [
        ("-h, --help", "Print help information"),
        ("-V, --version", "Print version"),
        ("-q, --quiet", "Suppress non-error output"),
        ("-v, --verbose", "Show extra details"),
        ("-f, --format <FMT>", "Output format: text, json, csv, toon"),
        ("--no-color", "Disable colored output"),
        ("--db <PATH>", "Path to database file"),
        ("--index <PATH>", "Path to search index directory"),
    ];

    for (opt, desc) in &options {
        output.print(&format!(
            "    {:24} {}",
            theme.format_info(opt),
            theme.format_muted(desc)
        ));
    }
    output.print("");

    // Examples section
    output.print(&theme.format_heading("EXAMPLES:"));
    let examples = [
        (
            "# Search for tweets about rust",
            "xf search \"rust programming\"",
        ),
        ("# Search only your DMs", "xf search \"meeting\" --types dm"),
        (
            "# Semantic search (finds related content)",
            "xf search \"feeling stressed\" --mode semantic",
        ),
        ("# Show archive statistics", "xf stats"),
        ("# Interactive search mode", "xf shell"),
        (
            "# Export results as JSON",
            "xf search \"project\" --format json",
        ),
    ];

    for (comment, command) in &examples {
        output.print(&format!("    {}", theme.format_muted(comment)));
        output.print(&format!("    {}", theme.format_primary(command)));
        output.print("");
    }

    // Footer
    output.print(
        &theme.format_muted("For more information: https://github.com/Dicklesworthstone/xf"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_as_str() {
        assert_eq!(Shell::Bash.as_str(), "bash");
        assert_eq!(Shell::Zsh.as_str(), "zsh");
        assert_eq!(Shell::Fish.as_str(), "fish");
        assert_eq!(Shell::PowerShell.as_str(), "powershell");
        assert_eq!(Shell::Elvish.as_str(), "elvish");
    }

    #[test]
    fn test_shell_config_file() {
        assert_eq!(Shell::Bash.config_file(), "~/.bashrc");
        assert_eq!(Shell::Zsh.config_file(), "~/.zshrc");
        assert!(Shell::Fish.config_file().contains("fish"));
        assert_eq!(Shell::PowerShell.config_file(), "$PROFILE");
        assert!(Shell::Elvish.config_file().contains("elvish"));
    }

    #[test]
    fn test_shell_install_instructions_not_empty() {
        for shell in [
            Shell::Bash,
            Shell::Zsh,
            Shell::Fish,
            Shell::PowerShell,
            Shell::Elvish,
        ] {
            let instructions = shell.install_instructions();
            assert!(
                !instructions.is_empty(),
                "{shell:?} should have instructions"
            );
            assert!(
                instructions.contains("xf completions"),
                "{shell:?} instructions should mention xf completions"
            );
        }
    }

    #[test]
    fn test_shell_to_clap_shell() {
        assert_eq!(Shell::Bash.to_clap_shell(), clap_complete::Shell::Bash);
        assert_eq!(Shell::Zsh.to_clap_shell(), clap_complete::Shell::Zsh);
        assert_eq!(Shell::Fish.to_clap_shell(), clap_complete::Shell::Fish);
        assert_eq!(
            Shell::PowerShell.to_clap_shell(),
            clap_complete::Shell::PowerShell
        );
        assert_eq!(Shell::Elvish.to_clap_shell(), clap_complete::Shell::Elvish);
    }

    #[test]
    fn test_shell_value_enum() {
        // Test that ValueEnum parsing works
        use clap::ValueEnum;
        assert_eq!(Shell::from_str("bash", true), Ok(Shell::Bash));
        assert_eq!(Shell::from_str("zsh", true), Ok(Shell::Zsh));
        assert_eq!(Shell::from_str("fish", true), Ok(Shell::Fish));
        assert_eq!(Shell::from_str("powershell", true), Ok(Shell::PowerShell));
        assert_eq!(Shell::from_str("elvish", true), Ok(Shell::Elvish));
    }

    #[test]
    fn test_render_styled_help_no_panic_tty() {
        let output = Output::new_tty();
        // Just verify no panic
        render_styled_help(&output);
    }

    #[test]
    fn test_render_styled_help_no_panic_piped() {
        let output = Output::new_piped();
        // Piped output falls back to clap help
        // Just verify no panic
        render_styled_help(&output);
    }
}

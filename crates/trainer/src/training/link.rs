use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::symlink as symlink_dir;
#[cfg(windows)]
use std::os::windows::fs::symlink_dir;

/// Refresh `<parent>/latest` to point at this run, so tools can open the newest run
/// without knowing its name. Best-effort: a link failure only warns.
pub fn refresh_latest(artifact_dir: &Path) {
    let Some(parent) = artifact_dir.parent() else {
        return;
    };
    let Some(name) = artifact_dir.file_name() else {
        return;
    };
    // Linking `latest` to itself would delete the run we just saved.
    if name == "latest" {
        return;
    }

    let link = parent.join("latest");
    if !replace_latest_link(&link) {
        return;
    }

    // A relative target keeps the link valid if the whole artifacts tree is moved.
    if let Err(error) = symlink_dir(name, &link) {
        tracing::warn!(
            link = %link.display(),
            target_dir = %Path::new(name).display(),
            %error,
            "could not refresh latest link; on Windows enable Developer Mode or pass \
             predict --artifact-dir explicitly"
        );
    }
}

/// Clear any existing `latest` link, returning whether the slot is free to link. A
/// real directory or file there is left untouched, so a misnamed run is never lost.
fn replace_latest_link(link: &Path) -> bool {
    // symlink_metadata inspects the link itself, not its target.
    let Ok(metadata) = link.symlink_metadata() else {
        return true;
    };

    if !metadata.file_type().is_symlink() {
        tracing::warn!(
            link = %link.display(),
            "latest slot exists and is not a symlink; leaving it untouched"
        );
        return false;
    }

    // A directory symlink unlinks with remove_dir on Windows, remove_file on Unix;
    // either removes only the link.
    #[cfg(windows)]
    let result = std::fs::remove_dir(link);
    #[cfg(unix)]
    let result = std::fs::remove_file(link);

    match result {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                link = %link.display(),
                %error,
                "could not remove existing latest link"
            );
            false
        }
    }
}

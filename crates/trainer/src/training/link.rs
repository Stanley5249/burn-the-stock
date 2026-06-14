use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::symlink as symlink_dir;
#[cfg(windows)]
use std::os::windows::fs::symlink_dir;

/// Refresh `<parent>/latest` to point at this run's artifact directory, so predict
/// and other tools can open the newest run without knowing its name. Best-effort:
/// the model and config are already on disk, so a link failure only warns rather
/// than failing the run.
pub fn refresh_latest(artifact_dir: &Path) {
    let Some(parent) = artifact_dir.parent() else {
        return;
    };
    let Some(name) = artifact_dir.file_name() else {
        return;
    };
    // Skip when the run already wrote into the `latest` slot, since linking it to
    // itself would only delete the run we just saved.
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

/// Clear any existing `latest` link so it can be recreated, returning whether the
/// slot is now free to link. A real directory or file there is left untouched and
/// reported rather than deleted, so a mistakenly named run is never lost.
fn replace_latest_link(link: &Path) -> bool {
    // symlink_metadata inspects the link entry itself rather than following it.
    let Ok(metadata) = link.symlink_metadata() else {
        // Nothing there yet, so the slot is free.
        return true;
    };

    if !metadata.file_type().is_symlink() {
        tracing::warn!(
            link = %link.display(),
            "latest slot exists and is not a symlink; leaving it untouched"
        );
        return false;
    }

    // A directory symlink unlinks with remove_dir on Windows but remove_file on
    // Unix; either removes only the link, never the target's contents.
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

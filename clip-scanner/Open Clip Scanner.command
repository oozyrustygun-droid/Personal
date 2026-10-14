#!/bin/zsh
# Double-click in Finder to start clip-scanner: it updates itself, then opens the scanner in your browser.
# Keep this Terminal window open while you scan; close it (or press Ctrl-C) to stop.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
echo "Checking for updates..."
git pull --ff-only --quiet || echo "(Couldn't update right now; starting the version you have.)"
exec uv run clip-scanner ui

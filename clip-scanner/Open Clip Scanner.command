#!/bin/zsh
# Double-click in Finder to start clip-scanner: it updates itself, then opens the scanner in your browser.
# Keep this Terminal window open while you scan; close it (or press Ctrl-C) to stop.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
echo "Checking for updates..."
if ! git pull --ff-only --autostash --quiet; then
  echo ""
  echo "!! Couldn't update the scanner (see the message above). Starting the version you already have."
  echo "!! Send a screenshot of this window to get it sorted out."
  echo ""
fi
echo "clip-scanner version: $(git log -1 --format='%h, %cs')"
exec uv run clip-scanner ui

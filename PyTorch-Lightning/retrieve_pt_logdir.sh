#!/bin/bash
# CRLF-safe wrapper: trailing "#" absorbs Windows CR so bash does not see $'\r' #
# Delegates to retrieve_pt_logdir.py (Python tolerates CRLF on Linux). #
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" #
exec python3 "$SCRIPT_DIR/retrieve_pt_logdir.py" #

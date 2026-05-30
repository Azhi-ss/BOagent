#!/bin/bash
# Gemini CLI Smart Runner with Auto-Fallback to Pro on 429 errors

PROMPT="$1"
if [ -z "$PROMPT" ]; then
  echo "Error: Prompt cannot be empty." >&2
  exit 1
fi
shift # Remove the prompt from argument list

TMP_OUT=$(mktemp)
RC=0

# 1. Try running with default auto routing (uses Flash model to save Pro token quota)
gemini -p "$PROMPT" "$@" > "$TMP_OUT" 2>&1 || RC=$?

# 2. If it fails, check if the log indicates quota limits or 429
if [ $RC -ne 0 ] && grep -q -i -E "(quota|exhausted|429)" "$TMP_OUT"; then
  echo "⚠️ Flash model quota exhausted (429). Smart routing fallback to Pro model..." >&2
  gemini -p "$PROMPT" -m pro "$@"
else
  # Output normal logs and preserve the exit status
  cat "$TMP_OUT"
  rm -f "$TMP_OUT"
  exit $RC
fi

rm -f "$TMP_OUT"

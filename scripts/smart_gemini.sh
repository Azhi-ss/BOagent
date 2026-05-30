#!/bin/bash
# Smart Gemini CLI Runner with Auto-Fallback to Pro on 429/quota errors.
# Usage: ./scripts/smart_gemini.sh "<prompt>" [extra_args...]

PROMPT="$1"
if [ -z "$PROMPT" ]; then
  echo "Error: Prompt cannot be empty." >&2
  exit 1
fi
shift # Remove prompt from arguments

TMP_OUT=$(mktemp)
RC=0

echo "Executing gemini prompt with auto model..." >&2
gemini -p "$PROMPT" --approval-mode yolo --skip-trust "$@" > "$TMP_OUT" 2>&1 || RC=$?

if [ $RC -ne 0 ] && grep -q -i -E "(quota|exhausted|429)" "$TMP_OUT"; then
  echo "⚠️ Flash model quota exhausted ($RC). Smart routing fallback to Pro model..." >&2
  gemini -m pro -p "$PROMPT" --approval-mode yolo --skip-trust "$@"
else
  cat "$TMP_OUT"
  rm -f "$TMP_OUT"
  exit $RC
fi

rm -f "$TMP_OUT"

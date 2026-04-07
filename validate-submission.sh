#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"

# --- Color Definitions ---
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# --- Configuration ---
DOCKER_BUILD_TIMEOUT=600

# --- Helper Functions ---
log() {
  printf "%b\n" "$1"
}

fail() {
  printf "${RED}✘ %b${NC}\n" "$1"
}

pass() {
  printf "${GREEN}✔ %b${NC}\n" "$1"
}

hint() {
  printf "  ${BOLD}Hint:${NC} %b\n" "$1"
}

stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %b${NC}\n" "$1"
  exit 1
}

run_with_timeout() {
  local time=$1
  shift
  timeout "$time" "$@"
}

# --- Step 1 ---
HF_URL=$1
REPO_DIR=$2

log "${BOLD}Step 1/3: Validating arguments${NC} ..."

if [ -z "$HF_URL" ] || [ -z "$REPO_DIR" ]; then
  fail "Missing arguments."
  hint "Usage: bash validate-submission.sh <HF_SPACE_URL> <REPO_DIR>"
  stop_at "Step 1"
fi

if [[ ! "$HF_URL" =~ ^https://.*hf\.space ]]; then
  fail "Invalid Hugging Face Space URL."
  hint "It should look like https://your-space.hf.space"
  stop_at "Step 1"
fi

if [ ! -d "$REPO_DIR" ]; then
  fail "Directory not found: $REPO_DIR"
  stop_at "Step 1"
else
  pass "Arguments are valid"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

#!/usr/bin/env bash
set -euo pipefail

exec python -m preprocessing.pipeline extract "$@"

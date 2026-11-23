#!/usr/bin/env bash
# Fast local quality checks before push
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== 1. Checking Ruff Linter ==="
.venv/bin/ruff check .

echo "=== 2. Checking Static JS Files ==="
node -c librarian/static/js/dashboard.js
node -c librarian/static/js/modules/index.js
node -c librarian/static/js/modules/api.js
node -c librarian/static/js/modules/state.js
node -c librarian/static/js/modules/utils.js
node -c librarian/static/js/modules/glossary.js

echo "=== 3. Running Core Unit Test Suite ==="
.venv/bin/pytest \
  tests/test_dashboard_api.py \
  tests/test_exporter.py \
  tests/test_cli.py \
  tests/test_template_assets.py \
  tests/test_audit_batch.py \
  tests/test_evaluator.py \
  tests/test_glossary_ledger.py \
  -q

echo "=== All quality checks PASSED cleanly! ==="

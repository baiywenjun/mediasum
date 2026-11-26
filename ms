#!/bin/bash
# mediasum 启动器：保证无论从哪个目录调用，都能找到包和虚拟环境。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/.venv/bin/python" -m mediasum "$@"

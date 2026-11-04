#!/usr/bin/env bash
# Run a command on ICE: infra/pace/ice.sh 'hostname; pace-quota'
# With no arguments, opens an interactive login.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec ssh -F "$here/ssh_config" ice "$@"

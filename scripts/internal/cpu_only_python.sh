#!/bin/sh
# Controlled comparison adapter: older runners do not forward this variable.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python "$@"

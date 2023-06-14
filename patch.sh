#!/usr/bin/env bash
set -euo pipefail

function patch_param() {
	sed -re "s/^($1) = .*/\1 = $2/"
}

cmd="python"
for arg in "$@"; do
	args=($arg)
	for arg in "${args[@]}"; do
		k="${arg%%=*}"
		v="${arg##*=}"
		cmd="patch_param $k $v | $cmd"
	done
done

< inference.py eval "$cmd"

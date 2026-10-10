#!/usr/bin/env bash
# Run a command with HUGGING_FACE_TOKEN set to a short-lived Hugging Face
# token, so CI stores no long-lived Hugging Face secret.
#
# The token comes from Hugging Face Trusted Publishers: the job's GitHub OIDC
# token is exchanged for a token scoped to the one repo named in
# HF_OIDC_RESOURCE (https://huggingface.co/docs/hub/trusted-publishers). It
# lasts 60 minutes from the exchange, so wrap each step that talks to Hugging
# Face separately rather than exporting one token for a long job. The job
# needs `permissions: id-token: write`, and the repo needs a publisher whose
# claims match this repository, branch and workflow file.
#
# Usage: HF_OIDC_RESOURCE=policyengine/policyengine-uk-data-private \
#          .github/with-hf-token.sh uv run --frozen make upload
set -euo pipefail
: "${HF_OIDC_RESOURCE:?HF_OIDC_RESOURCE must name the Hugging Face repo}"
if [ "$#" -eq 0 ]; then
  echo "::error::with-hf-token.sh needs a command to run" >&2
  exit 2
fi
# huggingface_hub 1.19.0 added the exchange; the project's own lock is older,
# so the CLI runs in its own throwaway environment.
token=$(uvx --from "huggingface_hub==2.2.0" hf auth token)
if [ -z "$token" ]; then
  echo "::error::The Hugging Face token exchange for ${HF_OIDC_RESOURCE} returned no token" >&2
  exit 1
fi
echo "::add-mask::${token}"
HUGGING_FACE_TOKEN="$token" exec "$@"

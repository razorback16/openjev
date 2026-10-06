# Use the original pinned serving image, not the experimental native image.
ARG B12X_BASE_IMAGE=vllm-node-b12x
FROM ${B12X_BASE_IMAGE}

# Existing serve.sh preflight delegates to the bridge verifier in this image.
ENV FORJEV_B12X_PROFILE=bridge
COPY pyproject.toml README.md /tmp/forjev-build/
COPY openjev /tmp/forjev-build/openjev
RUN python3 -m pip install --no-deps --no-build-isolation /tmp/forjev-build \
    && python3 -m openjev.b12x_logprobs_patch --apply \
    && rm -rf /tmp/forjev-build

# Inherit startup configuration; the existing launcher starts vLLM and its mods.

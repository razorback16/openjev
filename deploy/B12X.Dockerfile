# Build from the exact B12X image used by the serving launcher.
# The installer refuses unknown vLLM sources; no model weights are copied.
ARG B12X_BASE_IMAGE
FROM ${B12X_BASE_IMAGE}

COPY pyproject.toml README.md /tmp/forjev-build/
COPY openjev /tmp/forjev-build/openjev
RUN python3 -m pip install --no-deps --no-build-isolation /tmp/forjev-build \
    && python3 -m openjev.b12x_patch --apply \
    && rm -rf /tmp/forjev-build

# Inherit the base image's user, working directory, entrypoint and command.
# The existing launcher remains responsible for mods and starting vLLM.

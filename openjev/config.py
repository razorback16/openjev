"""Runtime settings, all from the environment."""
import os
from dataclasses import dataclass, field


def _env(name, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    upstream: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM", "http://127.0.0.1:8000"))
    upstream_model: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM_MODEL", "dgemma"))
    tokenizer: str = field(default_factory=lambda: _env("OPENJEV_TOKENIZER", "nvidia/diffusiongemma-26B-A4B-it-NVFP4"))
    backend: str = field(default_factory=lambda: _env("OPENJEV_BACKEND", "vllm"))
    mlx_model: str = field(default_factory=lambda: _env("OPENJEV_MLX_MODEL", "mlx-community/diffusiongemma-26B-A4B-it-4bit"))
    mlx_max_prompt: int = field(default_factory=lambda: int(_env("OPENJEV_MLX_MAX_PROMPT", "32768")))
    canvas: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS", "64")))
    canvas_step: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS_STEP", "16")))
    max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_INFLIGHT", "64")))
    max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_QUEUE", "512")))
    # Per-request bounds, so one body cannot fan out into unbounded work or memory.
    max_questions: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_QUESTIONS", "256")))
    max_body_bytes: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_BODY_BYTES", str(64 * 1024 * 1024))))
    # Seconds before a request passed through to another OpenJev container is a 503:
    # a heavy read (samples x groups, a long thought) can legitimately take minutes.
    forward_timeout: float = field(default_factory=lambda: float(_env("OPENJEV_FORWARD_TIMEOUT", "300")))
    # Optional auth. OPENJEV_API_KEY: clients send it as a Bearer token.
    # OPENJEV_ORIGIN_SECRET: a front proxy sends it as X-Origin-Secret.
    api_key: str = field(default_factory=lambda: _env("OPENJEV_API_KEY", ""))
    origin_secret: str = field(default_factory=lambda: _env("OPENJEV_ORIGIN_SECRET", ""))
    auto_threshold: float = field(default_factory=lambda: float(_env("OPENJEV_AUTO_THRESHOLD", "0.1")))
    auto_max: int = field(default_factory=lambda: int(_env("OPENJEV_AUTO_MAX", "4")))
    max_images: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGES", "8")))
    max_image_bytes: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))))
    # Kept small: generation denoises many blocks and must not crowd out System One reads.
    gen_max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_INFLIGHT", "8")))
    gen_max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_QUEUE", "32")))
    gen_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_TOKENS", "8192")))
    # The encoder backends (OPENJEV_BACKEND=laya, verdict, clm or jevk5). OPENJEV_DEVICE empty picks CUDA when present.
    laya_model: str = field(default_factory=lambda: _env("OPENJEV_LAYA_MODEL", "convaiinnovations/laya-typed-decisions"))
    verdict_model: str = field(default_factory=lambda: _env("OPENJEV_VERDICT_MODEL", "heman10x/rlcd-modernbert-151m"))
    device: str = field(default_factory=lambda: _env("OPENJEV_DEVICE", ""))
    encoder_batch: int = field(default_factory=lambda: int(_env("OPENJEV_ENCODER_BATCH", "16")))
    # CLM (OPENJEV_BACKEND=clm): its heads read Qwen3-8B embeddings from the vLLM server at OPENJEV_UPSTREAM.
    # OPENJEV_CLM_HEAD is a Hugging Face repo holding CLM_v0.1-8B.pt, or a local .pt file.
    clm_head: str = field(default_factory=lambda: _env("OPENJEV_CLM_HEAD", "Contrastive-LM/CLM-v0.1-8B"))
    clm_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_MAX_TOKENS", "2048")))
    clm_workers: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_WORKERS", "32")))
    clm_cache: str = field(default_factory=lambda: _env("OPENJEV_CLM_CACHE", "256MiB"))
    clm_embed_cache: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_EMBED_CACHE", "20000")))
    # JevK5 (OPENJEV_BACKEND=jevk5): the weights the vLLM server at OPENJEV_UPSTREAM serves. Their
    # jevk5_config.json holds the calibration temperature.
    jevk5_model: str = field(default_factory=lambda: _env("OPENJEV_MODEL", "alibiserikbay/JevK5"))
    jevk5_workers: int = field(default_factory=lambda: int(_env("OPENJEV_JEVK5_WORKERS", "32")))
    warmup: bool = field(default_factory=lambda: _env("OPENJEV_WARMUP", "1") != "0")
    # Other System One models served by other OpenJev containers: "name=url,name=url".
    # A request for one of them is passed through unchanged, so one origin serves all.
    model_routes: dict = field(default_factory=lambda: parse_routes(_env("OPENJEV_MODEL_ROUTES", "")))

    def __post_init__(self):
        """Refuse values a bad environment would otherwise turn into 500s or hangs:
        canvas_step=0 divides by zero on the first read, and a semaphore built with
        0 never opens, so every request would wait out its timeout instead of a 529."""
        positive = ("canvas", "canvas_step", "max_inflight", "max_questions", "max_body_bytes",
                    "max_image_bytes", "gen_max_inflight", "gen_max_tokens", "mlx_max_prompt",
                    "encoder_batch", "clm_workers", "jevk5_workers", "forward_timeout")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)!r} "
                                 f"(OPENJEV_{name.upper()})")
        for name in ("max_queue", "gen_max_queue", "max_images"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)!r} "
                                 f"(OPENJEV_{name.upper()})")


def parse_routes(text):
    routes = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, sep, url = part.partition("=")
        if not (sep and name.strip() and url.strip()):
            raise ValueError(f"OPENJEV_MODEL_ROUTES: {part!r} is not name=url")
        routes[name.strip()] = url.strip().rstrip("/")
    return routes


MODEL_VERSION = "openjev-0.1"
# openjev-0.1 is the wire name of this model, not the package version (see __init__.py).
MODEL_ALIASES = {"openjev-latest", MODEL_VERSION}
GEN_MODEL = "diffusiongemma-26b"
# accepted so TypeSafe's SDKs work unchanged (their default is jev-latest)
SDK_ALIASES = {"jev-latest", "jev-preview"}
MODELS = [
    {"name": "openjev-latest", "description": "Alias for the newest OpenJev release. Currently openjev-0.1.",
     "release_date": "2026-09-18"},
    {"name": "openjev-0.1", "description": "OpenJev 0.1: DiffusionGemma 26B-A4B (NVFP4) on vLLM's structured reads.",
     "release_date": "2026-09-18"},
    {"name": GEN_MODEL, "description": "DiffusionGemma 26B-A4B (NVFP4) text generation at POST /v1/chat/completions.",
     "release_date": "2026-09-18"},
]

# The encoder backends each serve one model, under its own name. The models are other
# people's work; the descriptions credit them wherever the model list is shown.
ENCODER_MODELS = {
    "laya": {"name": "laya-1.0",
             "description": "Laya by Nandakishor M / Convai Innovations (github.com/NandhaKishorM/laya, Apache-2.0): "
                            "the laya-typed-decisions checkpoint, a ModernBERT-large encoder (421M) fine-tuned on "
                            "the typed-decisions workflows. Text only, 1,024 tokens.",
             "release_date": "2026-09-22"},
    "verdict": {"name": "verdict-1.4",
                "description": "Verdict by Heman10x (github.com/Heman10x-NGU/Verdict-open-jev, Apache-2.0): "
                               "rlcd-modernbert-151m, a ModernBERT-base + GLiClass encoder (151M) calibrated with "
                               "RLCD, with the v1.4 inference engine. Text only, 512 tokens, up to 24 choices.",
                "release_date": "2026-09-22"},
    "clm": {"name": "clm-v0.1",
            "description": "CLM v0.1 by Contrastive-LM (github.com/Contrastive-LM/CLM, Apache-2.0): state and action "
                           "projection heads over Qwen3-8B last-token embeddings, here with Qwen3-8B-FP8 on vLLM. "
                           "Text only, 2,048 tokens.",
            "release_date": "2026-09-24"},
    "jevk5": {"name": "jevk5-0.2",
              "description": "JevK5 v0.2 by Alibi Serikbay (github.com/allebee/jevk5, Apache-2.0): Qwen3.5-4B with a "
                             "LoRA distilled from Qwen3.6-27B, merged, read as a softmax over the answer letters' "
                             "logits (SemIf's readout), on vLLM. Text only, 16,384 tokens.",
              "release_date": "2026-09-25"},
}


def served_models(backend):
    """(the model version a response names, the names a request may use, the /v1/models list)."""
    if backend in ENCODER_MODELS:
        m = ENCODER_MODELS[backend]
        return m["name"], {m["name"]} | SDK_ALIASES, [m]
    return MODEL_VERSION, MODEL_ALIASES | SDK_ALIASES, MODELS

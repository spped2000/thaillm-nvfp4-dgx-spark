"""Fix modelopt 0.43.0 --low_memory_mode crash with --attn_implementation.

patched_from_pretrained() forwards every model kwarg into accelerate's
load_checkpoint_and_dispatch(), which does not accept attn_implementation
(TypeError). The kwarg belongs to model construction, so route it into
cls.from_config() instead. Applied inside the disposable quant container only.
Run: python patch_modelopt_lowmem.py <path-to-modelopt-accelerate.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()

OLD_POP = 'trust_remote_code = kwargs.pop("trust_remote_code", False)'
OLD_FROM = "model = cls.from_config(config, torch_dtype=torch_dtype)"
OLD_QUANT = "mtq.quantize(model, quant_cfg)"

changed = []

# 1) attn_implementation belongs to from_config, not load_checkpoint_and_dispatch.
if "attn_implementation = kwargs.pop" not in src:
    if OLD_POP not in src or OLD_FROM not in src:
        print("PATCH FAILED: attn anchors not found — modelopt version drifted?")
        sys.exit(1)
    src = src.replace(
        OLD_POP,
        OLD_POP + '\n        attn_implementation = kwargs.pop("attn_implementation", None)',
    )
    src = src.replace(
        OLD_FROM,
        "model = cls.from_config(\n"
        "                config,\n"
        "                torch_dtype=torch_dtype,\n"
        "                **({'attn_implementation': attn_implementation} if attn_implementation else {}),\n"
        "            )",
    )
    changed.append("attn_implementation")

# 2) Tied weights (e.g. tie_word_embeddings) are absent from the checkpoint, so
# without tie_weights() they stay on meta and dispatch_model's .to() explodes.
# accelerate's own docs require tie_weights() before load_checkpoint_and_dispatch.
if "model.tie_weights()" not in src:
    if OLD_QUANT not in src:
        print("PATCH FAILED: mtq.quantize anchor not found")
        sys.exit(1)
    src = src.replace(OLD_QUANT, "model.tie_weights()\n        " + OLD_QUANT, 1)
    changed.append("tie_weights")

if not changed:
    print("already patched")
    sys.exit(0)
open(path, "w").write(src)
compile(src, path, "exec")
print("patched OK (" + ", ".join(changed) + "):", path)

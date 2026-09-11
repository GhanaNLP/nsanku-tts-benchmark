"""Recipe loader.

Each TTS model has a per-model recipe module `recipes/{owner}_{model}.py`
holding the knobs used to synthesise with it — guidance scale, inference
steps, and anything else that changes how it reads a sentence. The module is
real Python: edit it and open a pull request to change how that model is
benchmarked on the next run, rather than arguing with a number on a
leaderboard.

Models evaluated on several languages additionally get a recipe PER LANGUAGE:
`recipes/{owner}_{model}__{iso}.py`. That is where the per-language knobs live
— the API language code for hosted models, and the reference clip's transcript
for zero-shot models, which must be in the language being read. A contributor
can change one language without touching the others. Use
`load_lang_recipe(model_id, iso)` and `recipe_get(mod, "NAME", default)`.

A recipe may also define `build_wrapper(model_id, device, meta)` to take over
loading entirely.

A broken recipe (import error) does not kill the run: it logs a warning and
falls back to the built-in defaults.
"""
import importlib.util
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


def recipe_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}.py"


def lang_recipe_path(model_id: str, iso_code: str) -> Path:
    """Path of the per-language recipe for an API/LLM track."""
    safe = model_id.replace("/", "_").replace(":", "_")
    return RECIPES_DIR / f"{safe}__{iso_code}.py"


def load_recipe(model_id: str):
    """Import and return the model's recipe module, or None if it has none."""
    return _load(recipe_path(model_id))


def load_lang_recipe(model_id: str, iso_code: str):
    """Import the model's recipe for one language, or None if it has none."""
    return _load(lang_recipe_path(model_id, iso_code))


def recipe_get(mod, attr, default=None):
    """Value of `attr` from a recipe module, or `default` when it is not defined.

    An attribute the recipe explicitly sets to None wins over `default`, so a
    recipe can disable a language by setting `LANGUAGE_CODE = None`.
    """
    if mod is None or not hasattr(mod, attr):
        return default
    return getattr(mod, attr)


def _load(path: Path):
    if not path.exists():
        return None
    name = "nsanku_recipe_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"    WARN: recipe {path.name} failed to import ({e}); using defaults")
        return None
    return mod

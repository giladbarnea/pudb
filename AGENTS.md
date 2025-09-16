Purpose
- Orient future agents on why the stringifier works this way and what to preserve.

Why These Choices
- Do not parse repr strings. Instead, format from real objects to avoid brittleness.
- Global `max_length`. Users care about the final line width, not per-leaf limits.
- Two-phase rendering (coupled to tests and debug):
  - Phase 1: render without leaf-shortening. If len <= `max_length`, return as-is.
  - Phase 2: binary search a uniform leaf budget to approach (but stay <) `max_length`.
- Why binary search: final length grows monotonically with leaf budget; binary search finds the largest fitting budget in O(log N) evaluations, giving near-cap results without overfitting heuristics.
- Uniform leaf budget. Simpler, predictable, and keeps structure consistent across branches.
- Middle truncation for leaves (paths, strings, bytes) preserves both ends for recognition.
- No cycle-tracking for primitives. Prevents false "..." (e.g., repeated ints in dicts).

Coupling/Expectations
- `max_items` and `max_depth` apply in both phases; do not bypass them.
- Angle-bracket reprs and Enums:
  - Use full repr if it fits the leaf budget; otherwise shorten to class token (`<Type...>`).
- Paths use middle truncation only when needed to better utilize available budget.
- Pydantic models:
  - Use class-level field discovery (`model_fields` v2, `__fields__` v1) to avoid deprecation warnings.
  - `ValidationError` prints as a concise summary.

Debug/Dev Aids
- `-v` prints root input/output; `-vv` prints every call (two root invocations are expected: full-pass and budget-pass).
- Tests assert both content and length behavior (allowing a small margin to < `max_length`). Adjust margins only if the search strategy changes.

Handled Objects (summary)
- Primitives, str/bytes, pathlib paths, enums, angle-bracket reprs, dataclasses, namedtuples, mappings, list/tuple/set/frozenset, optional Pydantic BaseModel + ValidationError.

Extending Safely
- Add small, type-specific leaf formatters. Respect `_no_leaf_shorten` and `_leaf_max`.
- Preserve wrappers (quotes, (), [], {}, <>). Avoid parsing reprs.
- Keep top-level guarantee: final len(result) < `max_length`.

Handled Objects
- Primitives: `None`, bool, numbers.
- Text/bytes: `str` (keeps quotes), `bytes/bytearray` (keeps b-prefix).
- Paths: `pathlib.PurePath` (shortened path string, keeps type wrapper).
- Enums: prints `<EnumClass...>`.
- Callables/modules/angle-bracket reprs: shortened inside `<...>`.
- Dataclasses: `ClassName(field=value, ...)` via reflected fields.
- Namedtuples: `ClassName(field=value, ...)` via `_fields`.
- Mappings: `{key: value, ...}` (limited items).
- Sequences: `list`, `tuple` (singleton `(x,)`), `set`, `frozenset`.
- Optional Pydantic: `BaseModel` (v1/v2), `ValidationError` summary.

Approach
- Introspection-based recursive formatter; no parsing of `repr` strings.
- Shorten only leaves (strings/bytes/paths/angle-brackets) to keep structure.
- Cycle detection and limits for `max_depth` and `max_items`.
- Fallback: truncate `repr(obj)` when type is unknown.

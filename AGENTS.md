Purpose
- Defines the expected output contract and scope for `pudb_stringifier`.

Output Format
- Returns a single-line `str` representing the object.
- Preserves type wrappers and punctuation (quotes, (), [], {}, <>).
- Shortens inner content with `...` (middle-ellipsis), not at boundaries.
- Containers cap items with `...`; depth is limited; cycles print `...`.

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

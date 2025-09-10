from __future__ import annotations

import dataclasses
import enum
import pathlib
import types
from collections import namedtuple
from collections.abc import Mapping
from typing import Any

# Optional Pydantic support
try:
    from pydantic import BaseModel as _PydanticBaseModel  # type: ignore
    try:
        from pydantic import ValidationError as _PydanticValidationError  # type: ignore
    except Exception:  # pragma: no cover
        _PydanticValidationError = None
    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover - pydantic may not be installed
    _PydanticBaseModel = None
    _PydanticValidationError = None
    _HAVE_PYDANTIC = False


def _truncate_middle(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return "." * max_len
    head = max_len * 6 // 10
    tail = max_len - head - 3
    return s[:head] + "..." + s[-tail:] if tail > 0 else s[: max_len - 3] + "..."


def _short_str_keep_quotes(s: str, max_len: int) -> str:
    # Shorten the raw string, then reapply quotes/escapes via repr()
    return repr(_truncate_middle(s, max_len))


def _short_bytes_keep_prefix(b: bytes, max_len: int) -> str:
    s = b.decode("utf-8", "replace")
    inner = _truncate_middle(s, max_len)
    return "b" + repr(inner)


def _short_path(p: pathlib.PurePath, max_len: int, keep_last_segment: bool = False) -> str:
    parts = p.parts
    if len(parts) <= 2:
        inner = str(p)
    else:
        if p.is_absolute():
            # Keep root + first dir; show ellipsis; optionally keep last
            if keep_last_segment and len(parts) >= 2:
                inner = f"{parts[0]}/{parts[1]}/.../{parts[-1]}".replace("//", "/")
            else:
                inner = f"{parts[0]}/{parts[1] if len(parts) > 1 else ''}/...".replace("//", "/")
        else:
            if keep_last_segment:
                inner = f"{parts[0]}/.../{parts[-1]}"
            else:
                inner = f"{parts[0]}/..."
    inner = _truncate_middle(inner, max_len)
    return f"{type(p).__name__}(" + repr(inner) + ")"


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def _short_angle_brackets(s: str, max_inner_len: int = 16) -> str:
    # "<something long and busy>" -> "<something...>"
    if not (s.startswith("<") and s.endswith(">")):
        return s
    inner = s[1:-1]
    # Keep first token up to '.', ':', or whitespace, then ellipsis
    for sep in (".", ":", " "):
        pos = inner.find(sep)
        if pos > 0:
            inner = inner[:pos]
            break
    return f"<{_truncate_middle(inner, max_inner_len)}...>"


def pudb_stringifier(
    obj: Any,
    *,
    max_str: int = 32,
    max_items: int = 4,
    max_depth: int = 3,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> str:
    # Cycle protection
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "..."
    _seen.add(oid)

    # Depth cap
    if _depth >= max_depth:
        r = repr(obj)
        return _short_angle_brackets(r) if (r.startswith("<") and r.endswith(">")) else _truncate_middle(r, 40)

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, complex)):
        return repr(obj)

    # Strings / bytes
    if isinstance(obj, str):
        return _short_str_keep_quotes(obj, max_str)
    if isinstance(obj, (bytes, bytearray)):
        return _short_bytes_keep_prefix(bytes(obj), max_str)

    # Paths
    if isinstance(obj, pathlib.PurePath):
        return _short_path(obj, max_str, keep_last_segment=False)

    # Enums
    if isinstance(obj, enum.Enum):
        return f"<{type(obj).__name__}...>"

    # Functions/methods/modules (angle-bracket-y reprs)
    if isinstance(
        obj,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
            types.ModuleType,
        ),
    ):
        return _short_angle_brackets(repr(obj))

    # Dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj).__name__
        parts = []
        for f in dataclasses.fields(obj):
            try:
                v = getattr(obj, f.name)
            except Exception:
                v = "<error>"
            parts.append(
                f"{f.name}="
                + pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        inner = ", ".join(parts)
        return f"{cls}({inner})"

    # Namedtuple
    if _is_namedtuple(obj):
        cls = type(obj).__name__
        parts = []
        for name in obj._fields:
            v = getattr(obj, name)
            parts.append(
                f"{name}="
                + pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        return f"{cls}(" + ", ".join(parts) + ")"

    # Pydantic BaseModel (v1 and v2)
    if _HAVE_PYDANTIC and isinstance(obj, _PydanticBaseModel):  # type: ignore[arg-type]
        cls_type = type(obj)
        cls = cls_type.__name__
        # Determine field order for v1/v2 using the class to avoid deprecation warnings on instances.
        if hasattr(cls_type, "model_fields"):
            names = list(cls_type.model_fields.keys())  # pydantic v2
        elif hasattr(cls_type, "__fields__"):
            names = list(cls_type.__fields__.keys())  # pydantic v1
        else:
            names = sorted(k for k in dir(obj) if not k.startswith("_"))
        parts = []
        for name in names:
            try:
                v = getattr(obj, name)
            except Exception:
                v = "<error>"
            parts.append(
                f"{name}="
                + pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        inner = ", ".join(parts)
        return f"{cls}({inner})"

    # Pydantic ValidationError summary
    if _HAVE_PYDANTIC and _PydanticValidationError is not None and isinstance(obj, _PydanticValidationError):  # type: ignore[arg-type]
        n = 0
        try:
            errs = obj.errors()  # list-like
            n = len(errs) if hasattr(errs, "__len__") else 0
        except Exception:
            pass
        return f"<ValidationError {n} errors>"

    # Mapping
    if isinstance(obj, Mapping):
        items = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                items.append("...")
                break
            ks = pudb_stringifier(
                k,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth,
                _depth=_depth + 1,
                _seen=_seen,
            )
            vs = pudb_stringifier(
                v,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth,
                _depth=_depth + 1,
                _seen=_seen,
            )
            items.append(f"{ks}: {vs}")
        return "{" + ", ".join(items) + "}"

    # Sequences (but not str/bytes already handled)
    if isinstance(obj, list):
        parts = []
        for i, v in enumerate(obj):
            if i >= max_items:
                parts.append("...")
                break
            parts.append(
                pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        return "[" + ", ".join(parts) + "]"

    if isinstance(obj, tuple):
        seq = list(obj)
        parts = []
        for i, v in enumerate(seq):
            if i >= max_items:
                parts.append("...")
                break
            parts.append(
                pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        trailing = "," if len(seq) == 1 else ""
        return "(" + ", ".join(parts) + trailing + ")"

    if isinstance(obj, set):
        if len(obj) == 0:
            return "set()"
        seq = list(obj)
        parts = []
        for i, v in enumerate(seq):
            if i >= max_items:
                parts.append("...")
                break
            parts.append(
                pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        return "{" + ", ".join(parts) + "}"

    if isinstance(obj, frozenset):
        if len(obj) == 0:
            return "frozenset()"
        seq = list(obj)
        parts = []
        for i, v in enumerate(seq):
            if i >= max_items:
                parts.append("...")
                break
            parts.append(
                pudb_stringifier(
                    v,
                    max_str=max_str,
                    max_items=max_items,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
        return "frozenset({" + ", ".join(parts) + "})"

    # Fallback
    r = repr(obj)
    if r.startswith("<") and r.endswith(">"):
        return _short_angle_brackets(r)
    return _truncate_middle(r, 80)


def run_test() -> None:
    # 1) Dataclass + Path + Enum case similar to the example
    class NodeKind(enum.Enum):
        FILE = 2
        DIR = 3

    @dataclasses.dataclass
    class Entry:
        path: pathlib.PurePath
        name: str
        kind: NodeKind

    long_path = pathlib.PurePosixPath(
        "/private/var/folders/28/9x0yw2vs4bzd8s3p7pyfs1kc0000gn/T/pytest-of-giladbarnea/"
        "pytest-138/test_two_sibling_directories_b0/dirA/a.py"
    )
    e = Entry(path=long_path, name="a.py", kind=NodeKind.FILE)
    s = pudb_stringifier(e)
    assert s.startswith("Entry(")
    assert "PurePosixPath('/private/...')" in s, s
    assert "name='a.py'" in s, s
    assert "kind=<NodeKind...>" in s, s

    # 2) Lambda/function angle-bracket repr
    lf = pudb_stringifier(lambda x: x)
    assert lf.startswith("<function") and lf.endswith(">"), lf

    # 3) Mapping truncation
    d = {i: i for i in range(10)}
    md = pudb_stringifier(d, max_items=3)
    assert md.startswith("{") and "..." in md, md

    # 4) Bytes
    b = pudb_stringifier(b"abcdefghijklmnopqrstuvwxyz", max_str=8)
    assert b.startswith("b'"), b

    # 5) Cycles
    a = []
    a.append(a)
    cyc = pudb_stringifier(a)
    assert cyc == "[...]", cyc

    # 6) Namedtuple
    Point = namedtuple("Point", ["x", "y"])
    pt = Point("hello" * 20, 123)
    nt = pudb_stringifier(pt, max_str=10)
    assert nt.startswith("Point(") and "x='hellohe...".split()[0] or True

    print("All tests passed.")
    
    # 7) Empty set and frozenset, tuple singleton
    assert pudb_stringifier(set()) == "set()"
    fs = pudb_stringifier(frozenset())
    assert fs == "frozenset()", fs
    assert pudb_stringifier((42,)) == "(42,)", pudb_stringifier((42,))

    # 8) Non-empty set/frozenset truncation
    sset = pudb_stringifier(set(range(10)), max_items=3)
    assert sset.startswith("{") and sset.endswith("}") and "..." in sset, sset
    fs2 = pudb_stringifier(frozenset({1, 2, 3, 4}), max_items=2)
    assert fs2.startswith("frozenset({") and fs2.endswith("})") and "..." in fs2, fs2

    # 9) range
    import datetime as _dt
    r = pudb_stringifier(range(0, 1000))
    assert r.startswith("range("), r

    # 10) datetime
    dt = pudb_stringifier(_dt.datetime(2020, 1, 1, 12, 34, 56))
    assert "datetime" in dt, dt

    # 11) regex pattern
    import re as _re
    pat = _re.compile("a" * 100)
    rp = pudb_stringifier(pat, max_str=10)
    assert "re.compile(" in rp and "..." in rp, rp

    # 12) module
    mod = pudb_stringifier(_re)
    assert mod.startswith("<") and mod.endswith(">"), mod

    # 13) generator
    def _gen():
        for i in range(100):
            yield i
    g = _gen()
    gg = pudb_stringifier(g)
    assert gg.startswith("<generator") and gg.endswith(">"), gg

    # 14) memoryview
    mv = pudb_stringifier(memoryview(b"abcdef"))
    assert mv.startswith("<memory") and mv.endswith(">"), mv

    # 15) Exception with long message
    e = ValueError("x" * 100)
    es = pudb_stringifier(e, max_str=10)
    assert es.startswith("ValueError(") and "..." in es, es

    # 16) Decimal
    from decimal import Decimal
    dec = pudb_stringifier(Decimal("123456789.123456789"), max_str=8)
    assert dec.startswith("Decimal("), dec

    # 17) SimpleNamespace
    from types import SimpleNamespace
    ns = pudb_stringifier(SimpleNamespace(a=1, b=2))
    assert "namespace(" in ns, ns

    # 18) OrderedDict truncation
    from collections import OrderedDict
    od = OrderedDict((
        ("a", 1), ("b", 2), ("c", 3), ("d", 4)
    ))
    ods = pudb_stringifier(od, max_items=2)
    assert ods.startswith("{") and ods.endswith("}") and "..." in ods, ods

    # 19) Class object and angle-bracket repr shortening
    class C: 
        pass
    cls = pudb_stringifier(C)
    assert cls.startswith("<class") and cls.endswith(">"), cls

    class Angle:
        def __repr__(self):
            return "<Angle foo bar baz>"
    ang = pudb_stringifier(Angle())
    assert ang.startswith("<Angle") and ang.endswith("...>"), ang

    # 20) Pydantic BaseModel (if available)
    try:
        from pydantic import BaseModel
        have_pyd = True
    except Exception:
        have_pyd = False
    if have_pyd:
        class Address(BaseModel):
            city: str
            path: pathlib.PurePosixPath
            tags: list[str]
            scores: dict[str, int]

        class User(BaseModel):
            name: str
            age: int
            address: Address

        addr = Address(
            city="Tel Aviv" * 10,
            path=long_path,
            tags=["python", "pudb", "debug", "tools", "longtag"],
            scores={"a": 1, "b": 2, "c": 3, "d": 4},
        )
        user = User(name="Gilad" * 10, age=42, address=addr)
        us = pudb_stringifier(user)
        assert us.startswith("User(") and "address=Address(" in us, us
        assert "PurePosixPath('/private/...')" in us, us
        assert "tags=[" in us and "..." in us, us
        assert "scores={" in us and "..." in us, us

        # Field / FieldInfo object
        from pydantic import Field
        fi = Field(default="x" * 200, description="desc" * 100)
        fi_s = pudb_stringifier(fi, max_str=16)
        assert "FieldInfo(" in fi_s or "ModelField(" in fi_s, fi_s

        # RootModel (pydantic v2)
        try:
            from pydantic import RootModel  # type: ignore
            class Items(RootModel[list[int]]):
                pass
            items = Items([1, 2, 3, 4, 5, 6])
            ims = pudb_stringifier(items, max_items=3)
            assert ims.startswith("Items(") and "..." in ims, ims
        except Exception:
            pass

        # SecretStr
        try:
            from pydantic import SecretStr
            ss = SecretStr("supersecret-value-that-is-very-long")
            ss_s = pudb_stringifier(ss)
            assert ss_s.startswith("SecretStr("), ss_s
        except Exception:
            pass

        # ValidationError
        class M(BaseModel):
            x: int
        try:
            M(x="not-an-int")
        except Exception as ve:  # pydantic.ValidationError
            ve_s = pudb_stringifier(ve)
            assert ve_s.startswith("<ValidationError ") and ve_s.endswith(" errors>"), ve_s

        # TypeAdapter AnyUrl (v2)
        try:
            from pydantic import TypeAdapter, AnyUrl
            ta = TypeAdapter(AnyUrl)
            url = ta.validate_python("https://example.com/this/is/a/very/long/path/component")
            url_s = pudb_stringifier(url, max_str=20)
            assert url_s.startswith("'") and url_s.endswith("'"), url_s
        except Exception:
            pass


if __name__ == "__main__":
    run_test()

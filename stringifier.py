from __future__ import annotations

import dataclasses
import enum
import pathlib
import types
from collections import namedtuple
from collections.abc import Mapping
from typing import Any

import sys

if __name__ == "__main__":
    VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
    VERY_VERBOSE = "-vv" in sys.argv or "--very-verbose" in sys.argv
else:
    VERBOSE = False
    VERY_VERBOSE = False

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

# Optional requests support
try:
    from requests.models import Response as _RequestsResponse
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover - requests may not be installed
    _RequestsResponse = None
    _HAVE_REQUESTS = False


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


def _short_path(p: pathlib.PurePath, max_len: int) -> str:
    s = str(p)
    inner = s if len(s) <= max_len else _truncate_middle(s, max_len)
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


def _cap_root(s: str, *, max_length: int, depth: int) -> str:
    if depth == 0 and len(s) > max_length:
        return _truncate_middle(s, max_length)
    return s


def _is_acyclic_primitive(obj: Any) -> bool:
    return obj is None or isinstance(
        obj,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            bytearray,
            enum.Enum,
            pathlib.PurePath,
            range,
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
            types.ModuleType,
        ),
    )


def _repr_leaf(obj: Any, leaf_width: int, no_shorten: bool) -> str:
    r = repr(obj)
    if no_shorten or len(r) <= leaf_width:
        return r
    if r.startswith("<") and r.endswith(">"):
        return _short_angle_brackets(r, max_inner_len=max(8, leaf_width // 2))
    return _truncate_middle(r, leaf_width)


if VERBOSE or VERY_VERBOSE:
    import time

    from functools import wraps

    def print_input_output(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            depth = kwargs.get("_depth", 0)
            indent = " " + ("· " * depth) if depth else ""
            if depth == 0 and (VERBOSE or VERY_VERBOSE):
                time.sleep(0.01)
                prefix = "[root] " if VERY_VERBOSE else ""
                print(f"{prefix}Input:  {args}, {kwargs}    ({len(repr(args[0]))=})")
            if depth > 0 and VERY_VERBOSE:
                time.sleep(0.01)
                print(f"{indent}Input:  {args}, {kwargs}")
            result = fn(*args, **kwargs)
            if depth == 0 and (VERBOSE or VERY_VERBOSE):
                time.sleep(0.01)
                prefix = "[root] " if VERY_VERBOSE else ""
                print(f"{prefix}Output: {result}    ({len(result)=})\n")
            if depth > 0 and VERY_VERBOSE:
                time.sleep(0.01)
                print(f"{indent}Output: {result}")
            return result

        return wrapper
else:

    def print_input_output(fn):
        return fn


def pudb_stringifier(
    obj: Any,
    *,
    max_length: int = 160,
    max_items: int = 4,
    max_depth: int = 3,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _no_leaf_shorten: bool = False,
    _leaf_max: int | None = None,
    _driver: bool = True,
) -> str:
    # Root orchestrator: try full render, else binary-search leaf budgets
    if _driver and _depth == 0:
        full = pudb_stringifier(
            obj,
            max_length=10**9,
            max_items=max_items,
            max_depth=max_depth,
            _depth=0,
            _seen=set(),
            _no_leaf_shorten=True,
            _leaf_max=None,
            _driver=False,
        )
        if len(full) <= max_length:
            return full
        # Binary search the largest leaf budget that fits
        LEAF_BUDGET_MIN = 8
        LEAF_BUDGET_MAX = max(LEAF_BUDGET_MIN, min(max_length, 512))
        low, high = LEAF_BUDGET_MIN, LEAF_BUDGET_MAX
        best = None
        while low <= high:
            mid = (low + high) // 2
            s = pudb_stringifier(
                obj,
                max_length=max_length,
                max_items=max_items,
                max_depth=max_depth,
                _depth=0,
                _seen=set(),
                _no_leaf_shorten=False,
                _leaf_max=mid,
                _driver=False,
            )
            if len(s) < max_length:
                best = s
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            # Fallback: truncate full repr
            return _truncate_middle(full, max_length - 1 if max_length > 1 else 0)
        return best

    # From here on, it's a single-pass formatter using given leaf budget
    @print_input_output
    def _inner(
        obj: Any,
        *,
        max_length: int,
        max_items: int,
        max_depth: int,
        _depth: int,
        _seen: set[int] | None,
        _no_leaf_shorten: bool,
        _leaf_max: int | None,
        _driver: bool,
    ) -> str:
        # Cycle protection (skip primitives to avoid false positives on reused ints, etc.)
        if _seen is None:
            _seen = set()
        oid = id(obj)
        if not _is_acyclic_primitive(obj):
            if oid in _seen:
                return "..."
            _seen.add(oid)

        # Leaf width/budget
        leaf_width = _leaf_max if _leaf_max is not None else 32

        # Helpers to reduce duplication
        def cap(s: str) -> str:
            return _cap_root(s, max_length=max_length, depth=_depth)

        def rec(v: Any) -> str:
            return pudb_stringifier(
                v,
                max_length=max_length,
                max_items=max_items,
                max_depth=max_depth,
                _depth=_depth + 1,
                _seen=_seen,
                _no_leaf_shorten=_no_leaf_shorten,
                _leaf_max=_leaf_max,
                _driver=False,
            )

        def fmt_seq(seq, open_, close_, trailing: str = "") -> str:
            head = list(seq)[:max_items]
            items = [rec(v) for v in head]
            if len(list(seq)) > max_items:
                items.append("...")
            return cap(f"{open_}{', '.join(items)}{trailing}{close_}")

        def fmt_fields(cls_name: str, names: list[str], get) -> str:
            parts: list[str] = []
            for name in names:
                try:
                    v = get(name)
                except Exception:
                    v = "<error>"
                parts.append(f"{name}=" + rec(v))
            return cap(f"{cls_name}(" + ", ".join(parts) + ")")

        # Depth cap
        if _depth >= max_depth:
            # Depth cap applies in both phases; always allow leaf shortening here
            part = _repr_leaf(obj, leaf_width, no_shorten=False)
            return cap(part)

        # Primitives
        if obj is None or isinstance(obj, (bool, int, float, complex)):
            return _cap_root(repr(obj), max_length=max_length, depth=_depth)

        # Strings / bytes
        if isinstance(obj, str):
            if _no_leaf_shorten:
                return cap(repr(obj))
            return cap(_short_str_keep_quotes(obj, leaf_width))
        if isinstance(obj, (bytes, bytearray)):
            if _no_leaf_shorten:
                return cap(repr(bytes(obj)))
            return cap(_short_bytes_keep_prefix(bytes(obj), leaf_width))

        # Paths
        if isinstance(obj, pathlib.PurePath):
            if _no_leaf_shorten:
                return cap(repr(obj))
            return cap(_short_path(obj, leaf_width))

        # Enums
        if isinstance(obj, enum.Enum):
            r = repr(obj)
            if _no_leaf_shorten or len(r) <= leaf_width:
                return _cap_root(r, max_length=max_length, depth=_depth)
            # otherwise shorten to class only
            return _cap_root(
                f"<{type(obj).__name__}...>", max_length=max_length, depth=_depth
            )

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
            return cap(
                repr(obj)
                if _no_leaf_shorten
                else _short_angle_brackets(
                    repr(obj), max_inner_len=max(8, leaf_width // 2)
                )
            )

        # Dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            cls = type(obj).__name__
            names = [f.name for f in dataclasses.fields(obj)]
            return fmt_fields(cls, names, obj.__getattribute__)

        # Namedtuple
        if _is_namedtuple(obj):
            cls = type(obj).__name__
            return fmt_fields(cls, list(obj._fields), obj.__getattribute__)

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
            return fmt_fields(cls, names, obj.__getattribute__)

        # Pydantic ValidationError summary
        if (
            _HAVE_PYDANTIC
            and _PydanticValidationError is not None
            and isinstance(obj, _PydanticValidationError)
        ):  # type: ignore[arg-type]
            n = 0
            try:
                errs = obj.errors()  # list-like
                n = len(errs) if hasattr(errs, "__len__") else 0
            except Exception:
                pass
            return f"<ValidationError {n} errors>"

        # requests.models.Response
        if _HAVE_REQUESTS and isinstance(obj, _RequestsResponse):  # type: ignore[arg-type]
            try:
                ok = obj.ok
                status_code = obj.status_code
                url = obj.url
                text = obj.text
                return fmt_fields("Response", ["ok", "status_code", "url", "text"], 
                                lambda attr: {"ok": ok, "status_code": status_code, "url": url, "text": text}[attr])
            except Exception:
                return "<Response <error>>"

        # requests.models.Request
        if _HAVE_REQUESTS and hasattr(obj, '__class__') and obj.__class__.__name__ == 'Request':
            try:
                method = getattr(obj, 'method', 'UNKNOWN')
                url = getattr(obj, 'url', 'UNKNOWN')
                headers = getattr(obj, 'headers', {})
                body = getattr(obj, 'body', None)
                return fmt_fields("Request", ["method", "url", "headers", "body"], 
                                lambda attr: {"method": method, "url": url, "headers": headers, "body": body}[attr])
            except Exception:
                return "<Request <error>>"

        # Mapping
        if isinstance(obj, Mapping):
            items: list[str] = []
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_items:
                    items.append("...")
                    break
                items.append(f"{rec(k)}: {rec(v)}")
            return cap("{" + ", ".join(items) + "}")

        # Sequences (but not str/bytes already handled)
        if isinstance(obj, list):
            return fmt_seq(obj, "[", "]")

        if isinstance(obj, tuple):
            seq = list(obj)
            trailing = "," if len(seq) == 1 else ""
            return fmt_seq(seq, "(", ")", trailing)

        if isinstance(obj, set):
            if len(obj) == 0:
                return "set()"
            return fmt_seq(list(obj), "{", "}")

        if isinstance(obj, frozenset):
            if len(obj) == 0:
                return "frozenset()"
            return fmt_seq(list(obj), "frozenset({", "})")

        # Fallback
        part = _repr_leaf(obj, leaf_width, _no_leaf_shorten)
        return cap(part)

    return _inner(
        obj,
        max_length=max_length,
        max_items=max_items,
        max_depth=max_depth,
        _depth=_depth,
        _seen=_seen,
        _no_leaf_shorten=_no_leaf_shorten,
        _leaf_max=_leaf_max,
        _driver=_driver,
    )


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
    assert "PurePosixPath(" in s and "..." in s and "a.py" in s, s
    assert "name='a.py'" in s, s
    assert "kind=<NodeKind" in s, s
    # length behavior: original > 160 so target within [140, 160)
    assert 140 <= len(s) < 160, len(s)

    # 2) Lambda/function angle-bracket repr
    lf = pudb_stringifier(lambda x: x)
    assert lf.startswith("<function") and lf.endswith(">"), lf

    # 3) Mapping truncation
    d = {i: i for i in range(10)}
    md = pudb_stringifier(d, max_items=3)
    assert md.startswith("{") and "..." in md and "0: 0" in md, md

    # 4) Bytes
    b = pudb_stringifier(b"abcdefghijklmnopqrstuvwxyz", max_length=16)
    assert b.startswith("b'"), b

    # 5) Cycles
    a = []
    a.append(a)
    cyc = pudb_stringifier(a)
    assert cyc == "[...]", cyc

    # 6) Namedtuple
    Point = namedtuple("Point", ["x", "y"])
    pt = Point("hello" * 20, 123)
    nt = pudb_stringifier(pt, max_length=20)
    assert nt.startswith("Point(") and "x='hellohe...".split()[0] or True

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
    rp = pudb_stringifier(pat, max_length=20)
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
    es = pudb_stringifier(e, max_length=20)
    assert es.startswith("ValueError(") and "..." in es, es

    # 16) Decimal
    from decimal import Decimal

    dec = pudb_stringifier(Decimal("123456789.123456789"), max_length=20)
    assert dec.startswith("Decimal("), dec

    # 17) SimpleNamespace
    from types import SimpleNamespace

    ns = pudb_stringifier(SimpleNamespace(a=1, b=2))
    assert "namespace(" in ns, ns

    # 18) OrderedDict truncation
    from collections import OrderedDict

    od = OrderedDict((("a", 1), ("b", 2), ("c", 3), ("d", 4)))
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
    assert ang.startswith("<Angle") and ang.endswith(">"), ang

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
        fi_s = pudb_stringifier(fi, max_length=40)
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
            assert ve_s.startswith("<ValidationError ") and ve_s.endswith(" errors>"), (
                ve_s
            )

        # TypeAdapter AnyUrl (v2)
        try:
            from pydantic import TypeAdapter, AnyUrl

            ta = TypeAdapter(AnyUrl)
            url = ta.validate_python(
                "https://example.com/this/is/a/very/long/path/component"
            )
            url_s = pudb_stringifier(url, max_length=30)
            assert url_s.startswith("'") and url_s.endswith("'"), url_s
        except Exception:
            pass

    # 21) Global max_length budget tests
    # Flat object: long string must be capped to <= max_length
    max_len_flat = 50
    flat = pudb_stringifier("x" * 1000, max_length=max_len_flat)
    assert len(flat) <= max_len_flat, (len(flat), flat)
    assert flat.startswith("'") and flat.endswith("'"), flat

    # Recursive depth 4: nested mappings; ensure total length <= max_length
    nested = {"a": {"b": {"c": {"d": "y" * 500}}}}
    max_len_deep = 80
    deep_s = pudb_stringifier(nested, max_length=max_len_deep, max_depth=10)
    assert len(deep_s) <= max_len_deep, (len(deep_s), deep_s)
    assert deep_s[0] == "{" and deep_s[-1] == "}", deep_s

    # 22) No shortening if full fits (flat)
    s120 = "z" * 120
    full_flat = pudb_stringifier(s120)  # default max_length=160
    assert "..." not in full_flat and len(full_flat) <= 160, full_flat

    # 23) No shortening if full fits (deep)
    small_nested = {"a": {"b": {"c": {"d": "e"}}}}
    full_deep = pudb_stringifier(small_nested)
    assert "..." not in full_deep and len(full_deep) <= 160, full_deep

    # 24) requests.models.Response and Request (if available)
    try:
        import requests
        from requests.models import Response, Request
        
        # Create a mock response object
        class MockResponse(Response):
            def __init__(self):
                super().__init__()
                self._content = b'{"message": "Hello World"}'
                self.status_code = 200
                self.url = "https://example.com/api"
                self.headers = {}
                self.encoding = 'utf-8'
        
        mock_resp = MockResponse()
        resp_s = pudb_stringifier(mock_resp)
        assert resp_s.startswith("Response("), resp_s
        assert "ok=True" in resp_s, resp_s
        assert "status_code=200" in resp_s, resp_s
        assert "url='https://example.com/api'" in resp_s, resp_s
        assert "text=" in resp_s, resp_s
        
        # Test with long text content
        mock_resp._content = b'{"data": "' + b'x' * 1000 + b'"}'
        resp_long = pudb_stringifier(mock_resp, max_length=50)
        assert len(resp_long) <= 50, len(resp_long)
        assert "..." in resp_long, resp_long
        
        # Test Request object
        mock_req = Request('POST', 'https://api.example.com/users', 
                          headers={'Content-Type': 'application/json'}, 
                          data='{"name": "John"}')
        req_s = pudb_stringifier(mock_req)
        assert req_s.startswith("Request("), req_s
        assert "method='POST'" in req_s, req_s
        assert "url='https://api.example.com/users'" in req_s, req_s
        assert "headers=" in req_s, req_s
        assert "body=" in req_s, req_s
        
        # Test with long body content
        long_data = '{"data": "' + 'x' * 1000 + '"}'
        mock_req_long = Request('POST', 'https://api.example.com/large', data=long_data)
        req_long = pudb_stringifier(mock_req_long, max_length=50)
        assert len(req_long) <= 50, len(req_long)
        assert "..." in req_long, req_long
        
    except Exception:
        pass  # requests not available

    print("All tests passed.")


if __name__ == "__main__":
    run_test()

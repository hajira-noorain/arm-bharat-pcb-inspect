"""Minimal, lossless s-expression reader/writer for KiCad .kicad_pcb files (format versions 5 through 10).

Design: every character of the input is kept.  Each list node holds a flat token list where a token is either a
child ``Node`` or a ``str`` (an atom, a quoted string including its quotes, or a run of whitespace).
``dumps(parse(text)) == text`` for any text whose parentheses balance, so unrelated content is never touched
and an edited file differs from the original only where the editor changed it.

Only what the defect injector needs is provided: lookup (``find``/``findall``/``val``), edit (``set_val``,
``remove``, ``append``, ``insert_after``, ``replace``) and building new nodes from text (``parse_node``).
"""
from __future__ import annotations

import re
from typing import Iterator

_TOKEN = re.compile(r'\s+|\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+|.', re.S)


class Node:
    """A parenthesised list.  ``kids`` interleaves child Nodes, atoms and whitespace strings."""

    __slots__ = ("kids",)

    def __init__(self, kids: list | None = None):
        self.kids: list = kids if kids is not None else []

    # ---- reading -------------------------------------------------------------------------------
    @property
    def args(self) -> list:
        """Non-whitespace children (head atom first)."""
        return [k for k in self.kids if isinstance(k, Node) or not k.isspace()]

    @property
    def name(self) -> str:
        for k in self.kids:
            if isinstance(k, Node):
                return ""
            if not k.isspace():
                return k
        return ""

    def nodes(self) -> Iterator["Node"]:
        for k in self.kids:
            if isinstance(k, Node):
                yield k

    def findall(self, name: str) -> list["Node"]:
        return [k for k in self.kids if isinstance(k, Node) and k.name == name]

    def find(self, name: str) -> "Node | None":
        for k in self.kids:
            if isinstance(k, Node) and k.name == name:
                return k
        return None

    def val(self, name: str, default=None):
        """First argument of the first child ``(name arg ...)`` as unquoted text."""
        n = self.find(name)
        if n is None:
            return default
        a = n.args
        return unquote(a[1]) if len(a) > 1 and isinstance(a[1], str) else default

    def vals(self, name: str) -> list[str]:
        """All atom arguments of the first child ``(name a b c)`` as unquoted text (nested lists skipped)."""
        n = self.find(name)
        if n is None:
            return []
        return [unquote(a) for a in n.args[1:] if isinstance(a, str)]

    def floats(self, name: str) -> list[float]:
        out = []
        for v in self.vals(name):
            try:
                out.append(float(v))
            except ValueError:
                break
        return out

    # ---- editing -------------------------------------------------------------------------------
    def set_val(self, name: str, *values) -> None:
        """Replace the arguments of child ``(name ...)``; whitespace layout is kept when the arity matches."""
        n = self.find(name)
        if n is None:
            raise KeyError(name)
        idxs = [i for i, k in enumerate(n.kids) if isinstance(k, str) and not k.isspace()]
        txt = [fmt(v) if isinstance(v, float) else str(v) for v in values]
        arg_idx = idxs[1:]
        if len(arg_idx) == len(txt) and len(arg_idx) == len(n.args) - 1:
            for i, t in zip(arg_idx, txt):
                n.kids[i] = t
        else:
            n.kids = [n.kids[idxs[0]]] + [x for t in txt for x in (" ", t)]

    def remove(self, child: "Node") -> None:
        """Remove ``child`` together with the whitespace run in front of it."""
        for i, k in enumerate(self.kids):
            if k is child:
                j = i
                if j > 0 and isinstance(self.kids[j - 1], str) and self.kids[j - 1].isspace():
                    j -= 1
                del self.kids[j:i + 1]
                return
        raise ValueError("child not found")

    def append(self, child: "Node | str", sep: str = "\n  ") -> None:
        """Add ``child`` as the last element (before the closing parenthesis)."""
        if isinstance(child, str):
            child = parse_node(child)
        self.kids.append(sep)
        self.kids.append(child)

    def insert_after(self, ref: "Node", child: "Node | str", sep: str = "\n  ") -> None:
        if isinstance(child, str):
            child = parse_node(child)
        for i, k in enumerate(self.kids):
            if k is ref:
                self.kids[i + 1:i + 1] = [sep, child]
                return
        raise ValueError("ref not found")

    def replace(self, old: "Node", new: list["Node"]) -> None:
        """Replace ``old`` by one or more nodes, reusing the whitespace in front of ``old`` between them."""
        for i, k in enumerate(self.kids):
            if k is old:
                prev = self.kids[i - 1] if i > 0 else ""
                sep = prev if isinstance(prev, str) and prev.isspace() else "\n  "
                out: list = [new[0]]
                for n in new[1:]:
                    out += [sep, n]
                self.kids[i:i + 1] = out
                return
        raise ValueError("old not found")

    def copy(self) -> "Node":
        return Node([k.copy() if isinstance(k, Node) else k for k in self.kids])

    # ---- writing -------------------------------------------------------------------------------
    def write(self, out: list[str]) -> None:
        out.append("(")
        for k in self.kids:
            if isinstance(k, Node):
                k.write(out)
            else:
                out.append(k)
        out.append(")")

    def __str__(self) -> str:
        out: list[str] = []
        self.write(out)
        return "".join(out)

    def __repr__(self) -> str:
        s = str(self)
        return f"Node({s[:60]}{'...' if len(s) > 60 else ''})"


class Doc:
    """A parsed file: text before the root, the root node (``kicad_pcb``), text after it."""

    __slots__ = ("pre", "root", "post")

    def __init__(self, pre: str, root: Node, post: str):
        self.pre, self.root, self.post = pre, root, post

    def dumps(self) -> str:
        out = [self.pre]
        self.root.write(out)
        out.append(self.post)
        return "".join(out)


def parse(text: str) -> Doc:
    """Parse a whole file.  Raises ValueError on unbalanced parentheses or no root list."""
    stack: list[Node] = []
    root: Node | None = None
    pre: list[str] = []
    post: list[str] = []
    for m in _TOKEN.finditer(text):
        t = m.group(0)
        if t == "(":
            n = Node()
            if stack:
                stack[-1].kids.append(n)
            elif root is not None:
                raise ValueError("more than one top-level list")
            else:
                root = n
            stack.append(n)
        elif t == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            stack.pop()
        elif stack:
            stack[-1].kids.append(t)
        elif root is None:
            pre.append(t)
        else:
            post.append(t)
    if stack:
        raise ValueError("unbalanced '('")
    if root is None:
        raise ValueError("no s-expression found")
    return Doc("".join(pre), root, "".join(post))


def parse_node(text: str) -> Node:
    """Build a single node from text such as ``(segment (start 1 2) ...)``."""
    d = parse(text)
    if d.pre.strip() or d.post.strip():
        raise ValueError("extra text around node")
    return d.root


def dumps(doc: Doc) -> str:
    return doc.dumps()


def unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return re.sub(r"\\(.)", r"\1", s[1:-1])
    return s


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def fmt(x: float) -> str:
    """Number formatting for newly written values (nm resolution, trimmed)."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

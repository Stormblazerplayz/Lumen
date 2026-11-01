"""Lua 5.1 + LuaJIT syntax check and global-name report, with no Lua interpreter.

A recursive-descent parser that follows lparser.c (Lua 5.1 grammar, plus LuaJIT's goto/labels,
BeamNG's != for ~= and compound assignment (n += 1),
\\x and \\z escapes and LL/ULL/i number suffixes). It reports the first syntax error per file with
its line, and lists every name read or assigned that is not a local in scope: in BeamNG those
must be engine globals (map, be, vec3, extensions ...), so anything else is a typo or a missing
forward declaration.

usage: py luaparse.py [--globals] file.lua ...
"""
import re
import sys

KEYWORDS = {"and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if",
            "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while"}
SYMBOLS = ["..=", "...", "..", "==", "~=", "!=", "+=", "-=", "*=", "/=", "%=", "^=", "<=", ">=", "::", "+", "-", "*", "/", "%", "^", "#", "<", ">", "=",
           "(", ")", "{", "}", "[", "]", ";", ":", ",", "."]
BINARY = {"+": (6, 6), "-": (6, 6), "*": (7, 7), "/": (7, 7), "%": (7, 7), "^": (10, 9), "..": (5, 4),
          "==": (3, 3), "~=": (3, 3), "!=": (3, 3), "<": (3, 3), "<=": (3, 3), ">": (3, 3), ">=": (3, 3),
          "and": (2, 2), "or": (1, 1)}
UNARY_PRIORITY = 8
NUMBER = re.compile(r"0[xX][0-9a-fA-F]*(\.[0-9a-fA-F]*)?([pP][+-]?[0-9]+)?(ULL|ull|LL|ll|i|I)?|"
                    r"([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?(ULL|ull|LL|ll|i|I)?")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COMPOUND = {"+=", "-=", "*=", "/=", "%=", "^=", "..="}   # BeamNG LuaJIT: n += 1
LONG_COMMENT = re.compile(r"--\[(=*)\[")
LONG_STRING = re.compile(r"\[(=*)\[")


class LuaError(Exception):
    def __init__(self, line, text):
        super().__init__(f"line {line}: {text}")
        self.line = line


def tokenize(src):
    toks = []                       # (kind, value, line)
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if src.startswith("--", i):
            m = LONG_COMMENT.match(src, i)
            if m:
                close = "]" + m.group(1) + "]"
                j = src.find(close, i + len(m.group(0)))
                if j < 0:
                    raise LuaError(line, "unfinished long comment")
                line += src.count("\n", i, j)
                i = j + len(close)
            else:
                j = src.find("\n", i)
                i = n if j < 0 else j
            continue
        m = LONG_STRING.match(src, i)
        if m:
            close = "]" + m.group(1) + "]"
            j = src.find(close, i + len(m.group(0)))
            if j < 0:
                raise LuaError(line, "unfinished long string")
            toks.append(("string", src[i:j + len(close)], line))
            line += src.count("\n", i, j)
            i = j + len(close)
            continue
        if c in "'\"":
            j, start = i + 1, line
            while True:
                if j >= n:
                    raise LuaError(start, "unfinished string")
                d = src[j]
                if d == "\\":
                    e = src[j + 1] if j + 1 < n else ""
                    if e == "\n":
                        line += 1
                        j += 2
                    elif e == "z":
                        j += 2
                        while j < n and src[j] in " \t\r\n\f\v":
                            if src[j] == "\n":
                                line += 1
                            j += 1
                    elif e == "x":
                        if not re.match(r"[0-9a-fA-F]{2}", src[j + 2:j + 4]):
                            raise LuaError(line, "bad \\x escape")
                        j += 4
                    elif e.isdigit():
                        k = j + 1
                        while k < j + 4 and k < n and src[k].isdigit():
                            k += 1
                        if int(src[j + 1:k]) > 255:
                            raise LuaError(line, "escape sequence too large")
                        j = k
                    elif e in "abfnrtv\\\"'":
                        j += 2
                    else:
                        raise LuaError(line, f"invalid escape \\{e}")
                    continue
                if d == "\n":
                    raise LuaError(start, "unfinished string")
                if d == c:
                    break
                j += 1
            toks.append(("string", src[i:j + 1], start))
            i = j + 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = NUMBER.match(src, i)
            j = m.end()
            if j < n and (src[j].isalnum() or src[j] == "_"):
                raise LuaError(line, "malformed number near " + src[i:j + 1])
            toks.append(("number", src[i:j], line))
            i = j
            continue
        m = NAME.match(src, i)
        if m:
            w = m.group(0)
            toks.append(("keyword" if w in KEYWORDS else "name", w, line))
            i = m.end()
            continue
        for s in SYMBOLS:
            if src.startswith(s, i):
                toks.append(("symbol", s, line))
                i += len(s)
                break
        else:
            raise LuaError(line, f"unexpected character {c!r}")
    toks.append(("eof", "<eof>", line))
    return toks


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.p = 0
        self.scopes = [set()]
        self.globals = {}          # name -> (first line, "read"/"write")
        self.vararg = [True]       # main chunk is vararg

    # -- tokens --
    def peek(self, k=0):
        return self.t[self.p + k]

    def check(self, value):
        kind, v, _ = self.peek()
        return v == value and kind in ("keyword", "symbol")

    def accept(self, value):
        if self.check(value):
            self.p += 1
            return True
        return False

    def expect(self, value, what=None, opener=None):
        if not self.accept(value):
            kind, v, line = self.peek()
            extra = f" (to close {opener[0]!r} at line {opener[1]})" if opener else ""
            raise LuaError(line, f"{value!r} expected{extra} near {v!r}" + (f" ({what})" if what else ""))

    def name(self):
        kind, v, line = self.peek()
        if kind != "name":
            raise LuaError(line, f"<name> expected near {v!r}")
        self.p += 1
        return v, line

    # -- scopes --
    def declare(self, nm):
        self.scopes[-1].add(nm)

    def use(self, nm, line, mode):
        for s in reversed(self.scopes):
            if nm in s:
                return
        if nm not in self.globals:
            self.globals[nm] = (line, mode)

    def open_scope(self):
        self.scopes.append(set())

    def close_scope(self):
        self.scopes.pop()

    # -- blocks --
    def block_follow(self):
        kind, v, _ = self.peek()
        return kind == "eof" or (kind == "keyword" and v in ("else", "elseif", "end", "until"))

    def chunk(self):
        while not self.block_follow():
            if self.check("return"):
                self.p += 1
                if not self.block_follow() and not self.check(";"):
                    self.explist()
                self.accept(";")
                if not self.block_follow():
                    kind, v, line = self.peek()
                    raise LuaError(line, f"'end' expected after return near {v!r}")
                return
            if self.check("break"):
                self.p += 1
                self.accept(";")
                continue
            self.statement()
            self.accept(";")

    def block(self):
        self.open_scope()
        self.chunk()
        self.close_scope()

    def statement(self):
        kind, v, line = self.peek()
        if kind == "keyword":
            if v == "if":
                self.p += 1
                self.expr()
                self.expect("then", opener=("if", line))
                self.block()
                while self.check("elseif"):
                    l2 = self.peek()[2]
                    self.p += 1
                    self.expr()
                    self.expect("then", opener=("elseif", l2))
                    self.block()
                if self.accept("else"):
                    self.block()
                self.expect("end", opener=("if", line))
                return
            if v == "while":
                self.p += 1
                self.expr()
                self.expect("do", opener=("while", line))
                self.block()
                self.expect("end", opener=("while", line))
                return
            if v == "do":
                self.p += 1
                self.block()
                self.expect("end", opener=("do", line))
                return
            if v == "for":
                self.p += 1
                n1, _ = self.name()
                if self.accept("="):
                    self.expr()
                    self.expect(",")
                    self.expr()
                    if self.accept(","):
                        self.expr()
                    names = [n1]
                else:
                    names = [n1]
                    while self.accept(","):
                        names.append(self.name()[0])
                    self.expect("in", opener=("for", line))
                    self.explist()
                self.expect("do", opener=("for", line))
                self.open_scope()
                for nm in names:
                    self.declare(nm)
                self.block()
                self.close_scope()
                self.expect("end", opener=("for", line))
                return
            if v == "repeat":
                self.p += 1
                self.open_scope()           # 'until' sees the block's locals
                self.chunk()
                self.expect("until", opener=("repeat", line))
                self.expr()
                self.close_scope()
                return
            if v == "function":
                self.p += 1
                nm, l2 = self.name()
                method = False
                path = [nm]
                while self.accept("."):
                    path.append(self.name()[0])
                if self.accept(":"):
                    path.append(self.name()[0])
                    method = True
                self.use(nm, l2, "write" if len(path) == 1 else "read")
                self.funcbody(method, line)
                return
            if v == "local":
                self.p += 1
                if self.accept("function"):
                    nm, _ = self.name()
                    self.declare(nm)
                    self.funcbody(False, line)
                    return
                names = [self.name()[0]]
                while self.accept(","):
                    names.append(self.name()[0])
                if self.accept("="):
                    self.explist()
                for nm in names:
                    self.declare(nm)
                return
            if v == "goto":
                self.p += 1
                self.name()
                return
        if self.check("::"):
            self.p += 1
            self.name()
            self.expect("::")
            return
        # expression statement: a call, or an assignment
        kind_last = self.suffixedexp()
        if self.peek()[0] == "symbol" and self.peek()[1] in COMPOUND:
            if kind_last not in ("name", "index"):
                raise LuaError(line, "syntax error: cannot assign to this expression")
            self.p += 1
            self.expr()
            return
        if self.check("=") or self.check(","):
            if kind_last not in ("name", "index"):
                raise LuaError(line, "syntax error: cannot assign to this expression")
            while self.accept(","):
                k = self.suffixedexp(assign=True)
                if k not in ("name", "index"):
                    raise LuaError(line, "syntax error: cannot assign to this expression")
            self.expect("=")
            self.explist()
            return
        if kind_last != "call":
            kind2, v2, line2 = self.peek()
            raise LuaError(line2, f"syntax error near {v2!r} (a statement must be a call or an assignment)")

    def funcbody(self, method, line):
        self.open_scope()
        if method:
            self.declare("self")
        self.expect("(")
        vararg = False
        if not self.check(")"):
            while True:
                if self.accept("..."):
                    vararg = True
                    break
                self.declare(self.name()[0])
                if not self.accept(","):
                    break
        self.expect(")")
        self.vararg.append(vararg)
        self.chunk()
        self.vararg.pop()
        self.expect("end", opener=("function", line))
        self.close_scope()

    # -- expressions --
    def explist(self):
        self.expr()
        while self.accept(","):
            self.expr()

    def primaryexp(self, assign=False):
        kind, v, line = self.peek()
        if kind == "name":
            self.p += 1
            self.use(v, line, "write" if assign else "read")
            return "name"
        if self.accept("("):
            self.expr()
            self.expect(")", opener=("(", line))
            return "paren"
        raise LuaError(line, f"unexpected symbol near {v!r}")

    def suffixedexp(self, assign=False):
        # the first name in "a.b = 1" is read, not written: only a bare name is an assignment target
        start = self.p
        kind = self.primaryexp(assign=False)
        if assign or (self.check("=") or self.check(",")):
            pass
        while True:
            kind2, v, line = self.peek()
            if self.accept("."):
                self.name()
                kind = "index"
            elif self.check("["):
                self.p += 1
                self.expr()
                self.expect("]", opener=("[", line))
                kind = "index"
            elif self.accept(":"):
                self.name()
                self.callargs()
                kind = "call"
            elif self.check("(") or self.check("{") or kind2 == "string":
                if self.check("(") and self.t[self.p - 1][2] != line:
                    raise LuaError(line, "ambiguous syntax (function call x new statement)")
                self.callargs()
                kind = "call"
            else:
                break
        # a bare global name being assigned: record it as a write
        if kind == "name" and (self.check("=") or self.check(",") or assign):
            nm, line = self.t[start][1], self.t[start][2]
            if self.globals.get(nm, (0, ""))[1] == "read" and self.globals[nm][0] == line:
                self.globals[nm] = (line, "write")
        return kind

    def callargs(self):
        kind, v, line = self.peek()
        if kind == "string":
            self.p += 1
            return
        if self.check("{"):
            self.table()
            return
        self.expect("(")
        if not self.check(")"):
            self.explist()
        self.expect(")", opener=("(", line))

    def table(self):
        line = self.peek()[2]
        self.expect("{")
        while not self.check("}"):
            if self.check("["):
                l2 = self.peek()[2]
                self.p += 1
                self.expr()
                self.expect("]", opener=("[", l2))
                self.expect("=")
                self.expr()
            elif self.peek()[0] == "name" and self.peek(1)[1] == "=" and self.peek(1)[0] == "symbol":
                self.p += 2
                self.expr()
            else:
                self.expr()
            if not (self.accept(",") or self.accept(";")):
                break
        self.expect("}", opener=("{", line))

    def simpleexp(self):
        kind, v, line = self.peek()
        if kind in ("number", "string"):
            self.p += 1
            return
        if kind == "keyword" and v in ("nil", "true", "false"):
            self.p += 1
            return
        if self.check("..."):
            if not self.vararg[-1]:
                raise LuaError(line, "cannot use '...' outside a vararg function")
            self.p += 1
            return
        if self.check("{"):
            self.table()
            return
        if self.check("function"):
            self.p += 1
            self.funcbody(False, line)
            return
        self.suffixedexp()

    def expr(self, limit=0):
        kind, v, line = self.peek()
        if (kind == "keyword" and v == "not") or (kind == "symbol" and v in ("-", "#")):
            self.p += 1
            self.expr(UNARY_PRIORITY)
        else:
            self.simpleexp()
        while True:
            kind, v, line = self.peek()
            if kind not in ("symbol", "keyword") or v not in BINARY:
                return
            left, right = BINARY[v]
            if left <= limit:
                return
            self.p += 1
            self.expr(right)


def check_file(path):
    src = open(path, "rb").read().decode("utf-8", errors="replace")   # Lua reads bytes
    if src.startswith("﻿"):
        src = src[1:]
    p = Parser(tokenize(src))
    p.chunk()
    if p.peek()[0] != "eof":
        kind, v, line = p.peek()
        raise LuaError(line, f"'<eof>' expected near {v!r}")
    return p.globals


def main(argv):
    show = "--globals" in argv
    files = [a for a in argv if not a.startswith("--")]
    bad = 0
    for f in files:
        try:
            g = check_file(f)
            print(f"ok   {f}")
            if show:
                for nm in sorted(g, key=lambda k: g[k][0]):
                    line, mode = g[nm]
                    print(f"       {mode:5s} {nm:28s} line {line}")
        except LuaError as e:
            bad += 1
            print(f"FAIL {f}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

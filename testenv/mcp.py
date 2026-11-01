"""Tiny client for BeamNG's built-in MCP server (lua/ge/extensions/mcp/server.lua).

The game speaks JSON-RPC 2.0 over HTTP POST at http://127.0.0.1:<port>/mcp.
Used only against the isolated test instance started by game.py.

    py mcp.py ping [seconds]              wait until the server answers
    py mcp.py tools                       list tool names
    py mcp.py call run_lua '{"code":"return 1+1"}'
    py mcp.py lua "return be:getPlayerVehicleID(0)"
    py mcp.py shot out.png                take a screenshot, copy it to out.png
"""
import base64
import json
import os
import shutil
import socket
import sys
import time

PORT = int(os.environ.get("BNG_MCP_PORT", "29393"))
URL = f"http://127.0.0.1:{PORT}/mcp"
CRLF = "\r\n"
_next_id = [0]


class McpError(RuntimeError):
    pass


def _http_post(body, timeout):
    """Raw HTTP POST. BeamNG 0.39's server answers 'transfer-encoding: chunked' but
    writes the body unchunked (measured 2026-10-31 with curl: "chunk hex-length char
    not a hex digit"), which strict HTTP clients reject. So read the raw bytes and
    parse the JSON body ourselves."""
    head = CRLF.join([
        "POST /mcp HTTP/1.1",
        f"Host: 127.0.0.1:{PORT}",
        "Content-Type: application/json",
        "Accept: application/json",
        "Connection: close",
        f"Content-Length: {len(body)}",
        "", ""])
    with socket.create_connection(("127.0.0.1", PORT), timeout=timeout) as s:
        s.sendall(head.encode() + body)
        chunks = []
        t0 = time.time()
        while True:
            try:
                b = s.recv(65536)
            except socket.timeout:
                break
            if not b:
                break
            chunks.append(b)
            data = b"".join(chunks)
            _, sep, payload = data.partition(b"\r\n\r\n")
            if sep and payload:
                txt = payload.decode("utf-8", "replace")
                i, j = txt.find("{"), txt.rfind("}")
                if i >= 0 and j > i:
                    try:
                        return json.loads(txt[i:j + 1])
                    except ValueError:
                        pass  # incomplete, keep reading
            if time.time() - t0 > timeout:
                break
    raise McpError("no JSON in MCP reply: %r" % b"".join(chunks)[:200])


def rpc(method, params=None, timeout=30.0):
    _next_id[0] += 1
    body = json.dumps({"jsonrpc": "2.0", "id": _next_id[0], "method": method, "params": params or {}}).encode()
    data = _http_post(body, timeout)
    if "error" in data:
        raise McpError(f"{method}: {data['error']}")
    return data["result"]


def call(tool, args=None, timeout=60.0):
    """Call a tool. Returns text for text results, or (mimeType, bytes) for images.
    Raises McpError when the tool reports isError."""
    res = rpc("tools/call", {"name": tool, "arguments": args or {}}, timeout=timeout)
    content = res.get("content") or [{}]
    first = content[0]
    if first.get("type") == "image":
        return first.get("mimeType"), base64.b64decode(first["data"])
    text = first.get("text", "")
    if res.get("isError"):
        raise McpError(f"{tool}: {text}")
    return text


def lua(code, timeout=60.0):
    return call("run_lua", {"code": code}, timeout=timeout)


def wait_ready(max_s=600.0, poll=3.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < max_s:
        try:
            rpc("ping", timeout=5)
            return time.time() - t0
        except Exception as e:  # refused / empty while booting
            last = e
            time.sleep(poll)
    raise McpError(f"MCP not reachable on {URL} after {max_s:.0f}s: {last}")


def screenshot(dest):
    path = call("screenshot", {}).strip().strip('"')
    for _ in range(60):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            break
        time.sleep(0.25)
    time.sleep(0.3)
    shutil.copyfile(path, dest)
    return dest


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "ping":
        print("ready after %.1fs" % wait_ready(float(argv[2]) if len(argv) > 2 else 600))
    elif cmd == "tools":
        for t in rpc("tools/list")["tools"]:
            print(t["name"])
    elif cmd == "call":
        args = json.loads(argv[3]) if len(argv) > 3 else {}
        out = call(argv[2], args)
        print(out if isinstance(out, str) else f"<image {out[0]} {len(out[1])} bytes>")
    elif cmd == "lua":
        print(lua(argv[2]))
    elif cmd == "shot":
        print(screenshot(argv[2]))
    else:
        print("unknown command", cmd)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

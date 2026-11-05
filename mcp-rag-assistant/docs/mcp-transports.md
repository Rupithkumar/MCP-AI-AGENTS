# MCP transports

## stdio
With the stdio transport the client launches the server as a subprocess and exchanges messages over the standard input and output streams. This is the simplest option and is ideal for local tools, because no network port is opened and the server's lifetime is tied to the client. Because standard output carries the protocol messages, a stdio server must never print ordinary log lines to standard output. Logs should be written to standard error instead, otherwise they corrupt the message stream.

## Streamable HTTP
The Streamable HTTP transport runs the server as an independent web service with a single HTTP endpoint. Requests are sent with POST, and the server can answer with a stream of server-sent events when it needs to send several messages or progress updates. This transport supports remote servers that many clients share, and it supports authorization based on OAuth. Use it when a server must run on another machine or serve several users.

## The older HTTP+SSE transport
The earlier HTTP+SSE transport used two separate endpoints, one for posting requests and one for receiving events. It has been superseded by Streamable HTTP, which is simpler and handles reconnection better. New servers should use Streamable HTTP or stdio.

## Choosing a transport
Pick stdio for tools that run on the same computer as the host and need no authentication. Pick Streamable HTTP for remote, shared or multi-user servers. The two transports carry the same JSON-RPC messages, so the tools, resources and prompts of a server behave identically on either one.

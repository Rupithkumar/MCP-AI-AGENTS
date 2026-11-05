# Model Context Protocol overview

## What MCP is
The Model Context Protocol (MCP) is an open standard that Anthropic introduced in November 2024. It lets AI applications connect to external tools and data through one common protocol. Before MCP, every integration between a model and a service needed its own custom glue code. With MCP, a capability is built once as a server and then works with any compatible client.

## Architecture
MCP has three roles: the host, the client and the server. The host is the AI application the person talks to, such as a desktop assistant, an IDE or a custom agent. The host creates one client for each server it wants to use, and each client keeps a one-to-one connection with its server. The server is a lightweight program that exposes capabilities. All messages are encoded as JSON-RPC 2.0.

## Server primitives
A server can offer three kinds of capability. Tools are functions that the model can decide to call, for example running a query or creating an issue. Resources are read-only data that the application can load as context, such as files or database rows. Prompts are reusable templates that the user chooses explicitly, for example a "summarise my notes" command.

## Client features
Clients can offer features back to servers as well. Sampling lets a server ask the host's model for a completion, so the server does not need its own model access. Elicitation lets a server pause and ask the user for more input. Roots tell a server which parts of the file system it is allowed to work in.

## Discovery
When a client connects, it first performs an initialize handshake in which both sides announce their protocol version and capabilities. The client then calls tools/list to discover what tools exist, together with a JSON schema for each tool's arguments. Because tools are discovered at runtime, adding a new server gives an agent new abilities without changing the agent's code.

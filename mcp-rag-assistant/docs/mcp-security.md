# Security of MCP servers and agents

## Main risks
Prompt injection happens when text returned by a tool or document contains instructions that the model mistakes for commands from the user. Tool poisoning is a related attack in which a malicious server hides instructions inside a tool's description, which the model reads whenever it lists the available tools. An over-privileged server is a risk on its own: if a tool can read any file or run any command, one mistaken call can do real damage. Server code can also contain classic bugs such as command injection.

## File access and path traversal
A tool that reads files must stay inside an allowed folder. The safe pattern is to join the requested path to the sandbox root, resolve it to an absolute path, and then check that the result is still inside the root. A request such as ../../etc/passwd resolves outside the root and must be rejected. Checking the string for ".." is not enough, because symbolic links and encoded paths can bypass it.

## Never evaluate untrusted input
A calculator tool should never pass the model's expression to eval. Parse it into a syntax tree instead and allow only numbers and arithmetic operators. Anything else, such as a function call or an import, is rejected before it can run.

## Mitigations
Apply least privilege: give each server only the access it needs. Require human approval for destructive actions such as deleting data or sending messages, so a mistaken tool call cannot cause permanent harm. Validate every argument against its schema and return clear error messages that the model can read and correct. Keep an audit log of tool calls, pin the versions of third-party servers, and treat everything that comes back from a tool as untrusted data, never as instructions.

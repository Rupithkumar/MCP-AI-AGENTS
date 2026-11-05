# The agent loop

## Reason, act, observe
An AI agent is a language model running in a loop. In each turn the model reasons about the task, decides whether to call a tool, and receives the tool's result as an observation. The loop repeats until the model answers without requesting another tool call. This is why the same model can chain several tools together: it sees each result before choosing the next step.

## Step limits
Every agent needs a hard cap on the number of steps, because a confused model can call tools forever. A typical limit is 8 to 10 steps per task. When the cap is reached, the agent stops and reports that it could not finish, rather than looping and spending money.

## Handling tool errors
When a tool fails, the error should be returned to the model as readable text, flagged as an error, instead of crashing the program. The model can then read the message, correct its arguments and try again. Unknown tool names and malformed arguments should be handled the same way.

## Keeping the context under control
Each step adds the tool call and its result to the conversation, so the context grows quickly. Large tool outputs should be truncated before they are sent back to the model, for example to 8,000 characters, and the full data can be kept elsewhere. Smaller contexts are cheaper and also help the model stay focused.

## Human approval
Some tools change the world, for example deleting a record. The agent can pause before running such a tool and ask the person to approve it. If the person declines, the denial is returned to the model as a tool result so it can explain the situation instead of retrying.

---
audience: public
items:
  - tool:read_draft
---

# The assistant's tools

The assistant in a chat can do more than reply with text. It can use tools to look at the work you are doing together. This page explains the one tool described here, `read_draft`, so you know what the assistant can see and when it uses it.

## Goal

By the end of this lesson you will know:

- what `read_draft` does,
- what argument it takes,
- how to refer to a draft so the assistant can read it,
- what to watch out for.

## Prerequisites

- You have a chat open with the assistant.
- The chat has produced at least one draft. A draft is a configuration written in TOML.

## What `read_draft` does

`read_draft` reads a draft of this chat (its TOML). The assistant uses it to check or explain a draft before revising it. Reading the draft first means the assistant works from what is actually there, not from memory of what it wrote earlier.

### Argument

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `draft` | string | yes | The draft as the chat named it, for example `UI/<you>/drafts/...`. |

No other arguments are accepted. `draft` is the only one, and you must provide it.

## Steps

1. Ask the assistant to check, explain or revise a draft from this chat.
2. Refer to the draft by the name the chat gave it. The name looks like `UI/<you>/drafts/...`.
3. The assistant calls `read_draft` with that name as `draft`.
4. The assistant answers from the TOML it read. If you asked for a revision, it revises that text.

## Worked example

Suppose the chat has produced a draft and named it `UI/<you>/drafts/...`. You write:

> Please explain the draft `UI/<you>/drafts/...` and tell me whether anything looks wrong before you change it.

The assistant calls `read_draft` with this argument:

```json
{"draft": "UI/<you>/drafts/..."}
```

The tool returns the draft's TOML. The assistant then walks you through it section by section and points out anything that needs attention. When you approve, it revises the draft.

Note that `...` stands for the rest of the name. Use the full name exactly as the chat showed it.

## Pitfalls

- **Use the name the chat gave.** `draft` must be the draft as the chat named it. A name you made up, or a shortened one, may not match.
- **Only drafts of this chat.** `read_draft` reads a draft of this chat. Do not expect it to read drafts from other chats.
- **Reading is not changing.** `read_draft` only reads. Revising is a separate step that you ask for.
- **Do not add arguments.** Only `draft` is accepted.

## What to read next

- The other pages of this guide, to see how the settings in a draft's TOML are meant to be used.

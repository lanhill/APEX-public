---
audience: public
---

# The APEX user guide

This guide teaches APEX by doing. Each lesson states its goal and what you need, walks through the steps, works an example, and lists the pitfalls and what to read next.

## What APEX is, and what a plugin is

APEX is the application this guide documents. You work with it through its command-line tool, `apex`, through its configuration, and through its web UI.

A **plugin** is a building block that gives APEX one capability. Each plugin has **options**, which are named settings with a type and a default value. You set options in APEX's configuration to change how the plugin behaves. Many lessons are about one plugin and its options.

## How the lessons are organised

The lessons follow APEX itself, in three groups, with a reference beside them:

1. **Plugins and their options**: what each plugin does and how to set its options.
2. **Commands**: the `apex` commands and their flags.
3. **The web UI**: how to do things from the browser.

How settings are written down is in the reference: its configuration pages list every section and option of each kind of configuration file, for example [the history-matching configuration](../reference/configuration/history_match.md) and [the optimization configuration](../reference/configuration/optimization.md). `apex schema KIND` prints the same as a JSON Schema.

## Where to begin

Read the groups in the order above. Start with the first lesson on plugins and their options. It introduces the terms the other groups rely on: plugin, option, type and default. After that, move on to commands, then the web UI, and look up configuration sections in the reference as you need them. Once you know the terms, you can also jump straight to the group that matches your task.

## Pitfalls

- Do not skip the vocabulary. Later lessons use the words *plugin* and *option* without explaining them again.
- Copy examples exactly. Every example is checked against APEX before a lesson is published, and a lesson is revised when what it teaches changes. An example you have altered may no longer behave as described.

## What to read next

Open the first lesson in the plugins and options group, [executors](plugins/executors.md), and work through it from its goal to its worked example.

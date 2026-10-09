# Removed Command

`/awos:roadmap` was removed from AWOS. AWOS is not a planning or backlog tool: specs are anchored by an explicit topic now, not by a roadmap item. This notice replaced the project's local copy of the old command so that calling it says so instead of running a frozen 1.x copy.

Tell the user the following, then stop. Do not build, update, or review a roadmap document.

- This command no longer ships with AWOS. To start a feature, run `/awos:spec` and state the topic directly.
- If `context/product/roadmap.md` exists, it is theirs: AWOS no longer creates, updates, or marks it, so keeping it current is their own work now — by hand or with a command of their own. While it exists, the spec command still offers its incomplete items as topic candidates.
- To remove this notice, delete `.claude/commands/awos/roadmap.md` and `.awos/commands/roadmap.md`.

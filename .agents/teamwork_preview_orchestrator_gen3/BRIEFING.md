# BRIEFING — 2026-09-23T02:10:00Z

## Mission
Deliver Milestones M3 (Non-IID IoT Benchmark Harness), M4 (Remote Execution & Git Branch), and M5 (Walkthrough Documentation & Verification) for Federated LUNAR.

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_gen3
- Original parent: parent
- Original parent conversation ID: 997be87d-2cd2-417f-802c-33722df71245

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\teamwork_preview_orchestrator_1\PROJECT.md
1. **Decompose**: Milestones M1 to M5
2. **Dispatch & Execute**: Direct iteration loop (Explorer -> Worker -> Reviewer -> Challenger -> Auditor -> Gate)
3. **On failure**: Retry -> Replace -> Skip -> Redistribute -> Redesign
4. **Succession**: Self-succeed at 16 spawns
- **Work items**:
  1. Milestone M1: Core Fed-LUNAR Engine [done]
  2. Milestone M2: 3-Tier Baseline Hierarchy [done]
  3. Milestone M3: Non-IID Dirichlet IoT Benchmark Harness & Metrics [in-progress]
  4. Milestone M4: Remote Execution on postmaster.iec & Git Branch [pending]
  5. Milestone M5: Walkthrough Documentation & Final Verification [pending]
- **Current phase**: Milestone M3
- **Current focus**: Milestone M3 (Non-IID Dirichlet IoT Benchmark Harness across BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT in `fed_lunar/benchmark/`)

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- You MAY use file-editing tools ONLY for metadata/state files (.md) in your .agents/ folder.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.
- Always include path to ORIGINAL_REQUEST.md in every dispatch.
- Mandatory integrity warning in Worker prompts.

## Current Parent
- Conversation ID: 997be87d-2cd2-417f-802c-33722df71245
- Updated: 2026-09-23T02:08:56Z

## Key Decisions Made
- Milestone M2 Gate certified PASS after remediation (37/37 unit tests and 126/126 E2E tests passing).
- Advancing directly to Milestone M3 execution.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_m3_1 | teamwork_preview_explorer | Dataset structure & loader design | completed | af85ffb4-afd3-4701-851c-c873bf18cf84 |
| explorer_m3_2 | teamwork_preview_explorer | Non-IID Dirichlet & runner design | completed | 61a60469-19db-4fb4-8288-1c2c36157782 |
| explorer_m3_3 | teamwork_preview_explorer | Metrics & test harness design | completed | 95106087-4139-457a-8b3f-650d8484429f |
| worker_m3_1 | teamwork_preview_worker | M3 benchmark harness & tests implementation | in-progress | c6ec0e08-9d55-4693-b840-eafa2f3adb1b |

## Succession Status
- Succession required: no
- Spawn count: 7 / 16
- Pending subagents: c6ec0e08-9d55-4693-b840-eafa2f3adb1b
- Predecessor: teamwork_preview_orchestrator_1
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 6d043925-dcfa-4d4b-9fed-52cd89b43248/task-28
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run manage_task(Action="list") — re-create if missing

## Artifact Index
- ORIGINAL_REQUEST.md — User requirements
- PROJECT.md — Global architecture, features, milestones, interfaces, code layout
- GATE_STATUS.md — Gate verdicts per milestone
- TEST_READY.md — 126 E2E tests passing

Capability snapshot of Leading Platform vs CVAT Community
Leading Platform’s documented strengths relevant to your workflow:

Multimodal labeling across image, video, audio (and other modalities) and “multimodal annotation in one view,” including customizable annotation layouts.
Video labeling with temporal constructs, including applying classifications over frame ranges via explicit “set start,” “set end,” and “add range” controls/hotkeys.
Audio labeling with time-region constructs (regions applied to millisecond ranges) and audio metadata handling.
Customizable workflows (canvas-based) with stages, routers, and workflow webhooks (e.g., trigger when a task reaches “Complete”).
Consensus workflows where multiple annotators label independently and reviewers compare/choose labels (bulk or individual).
Security posture claims (HIPAA/SOC 2) and storage-access patterns like signed URLs and “strict client-only access” options for certain integrations.
CVAT Community’s relevant baseline capabilities (important because some parity work is already done):

Image/video/3D labeling focus: CVAT is framed as a platform for “visual datasets” and provides image/video/3D annotation plus QA, automation, and team collaboration.
Role model & policy control: user roles exist; for self-hosted deployments roles can be adjusted via OPA .rego policy files under cvat/apps/*/rules/.
Workflow primitives: jobs have stages annotation, validation, acceptance, switchable via UI, and an assignee performs stage-specific work.
Manual QA / issue reporting in Review mode.
Automated QA via ground-truth jobs / honeypots (2D) and quality analytics.
Consensus-based annotation: CVAT can create “consensus replica jobs,” merge annotations, and produce consensus scores/votes for review; however consensus is limited to tasks/jobs and explicitly “projects are not supported.”
Cloud storage attachment for importing/exporting datasets (S3/Azure Blob/GCS/Backblaze B2).
Webhooks triggered by events (e.g., job status updates) and configurable via GUI or API.
Repository modularity (useful for “agentic SWE engineers” to know where to work): CVAT docs identify top-level packages like cvat-ui/, cvat-sdk/, cvat-data/ (media decoding), and serverless/ (AI models).

__

Explicit feature list: what CVAT Community is missing (or only partially covers) that you would likely need to implement
The checklist below is intentionally written as a backlog spec for a SWE team building a CVAT fork. Each item includes (a) how Leading Platform represents the feature, (b) why CVAT Community’s baseline is insufficient for the surgeon workflow, and (c) build notes suitable for breaking down into engineering tasks.

Priority legend:

P0 = must-have to meet the core project requirements (narration + temporal labels + surgeon UX)
P1 = required to scale to 20–50 surgeons without PM pain
P2 = “parity polish” / leverage if you want Leading Platform-like operational smoothness

____

## P0 checklist (core workflow)

- [ ] Temporal range labeling for phases/subtasks/gestures (interval annotations)
  - Leading Platform: Apply classifications over frame ranges via explicit “set start,” “set end,” “add range” controls/hotkeys.
  - CVAT Community gap: “Tags” are frame-based; common workaround is tagging the first frame and post-processing.
  - Build notes:
    - Add a first-class `IntervalTag`/`IntervalClassification` type storing `[start_frame, end_frame]` + label/attributes.
    - Add timeline UI to create/resize/lock intervals, snap to frame boundaries, and show stacked phase tracks.
    - Export intervals directly; optionally render to per-frame tags for legacy exporters.

- [ ] Multimodal “video + narration” workspace
  - Leading Platform: “Multimodal annotation in one view” with customizable multimodal layouts.
  - CVAT Community gap: UI and docs focus on “visual datasets” rather than multimodal (audio+video) workspaces.
  - Build notes:
    - Implement a multimodal panel system in the Job UI: video player + timeline + “Narration” tool + “Phase track” editor.
    - Add a “Procedure profile” to lock the layout and reduce surgeon cognitive load.

- [ ] Audio narration capture (record, review, re-record) with time sync to video
  - Leading Platform: Audio modality support with time-region constructs down to millisecond ranges.
  - CVAT Community gap: No native audio waveform/timestamp labeling editor or audio annotation constructs in core.
  - Build notes:
    - Add in-browser recording (WebAudio `getUserMedia`) while watching/controlling video playback.
    - Persist audio as a job/task artifact with alignment metadata (`video_time_offset`, `sample_rate`, `start_wallclock`, etc.).
    - Add waveform visualization and timecode markers tied to video frames.

- [x] Narration-to-text pipeline hooks (for reasoning dataset distillation)
  - Leading Platform: Multimodal workflows + automation/webhooks + programmatic label access.
  - CVAT Community gap: Webhooks exist, but there are no narration artifacts or narration-specific triggers by default.
  - Build notes:
    - Extend the event model to emit `narration_uploaded`, `narration_finalized`, `transcript_ready`.
    - Route via CVAT webhooks (or an internal event bus) to trigger transcription + downstream dataset generation.
    - Keep PHI boundaries in mind (controlled storage for audio/transcripts).
  - Status: `JobTranscript` model (raw + corrected text, word timestamps, status, model info) with migration 0102. Two-stage pipeline worker: Parakeet-TDT-0.6B-v2 ASR → LLM post-correction with surgical context from video-level classifications. Auto-triggered via `post_save` signal on `JobNarration` → RQ job. Webhook events added: `create:narration`, `create:transcript`, `update:transcript`. `TranscriptViewer` component in surgery sidebar Narration tab with live polling during processing. GET /jobs/{id}/transcripts API. Configurable via Django settings: `TRANSCRIPTION_ASR_URL`, `TRANSCRIPTION_LLM_URL`, `TRANSCRIPTION_LLM_MODEL`.

- [x] Surgeon-first “Queue” UX (one click: start next assigned task/job)
  - Leading Platform: A “Queue” tab to start work quickly; can hide unassigned tasks so annotators only see their queue.
  - CVAT Community gap: Navigation is project/task/job browsing; stages are not role-gated and are easy to flip accidentally.
  - Build notes:
    - Add a top-level “My Work” route showing only assigned jobs requiring surgeon action, with Start/Resume/Next.
    - Enforce restrictions via OPA policies and/or UI gating to prevent browsing other data.
  - Status: `/my-work` route with `MyWorkPage` component. Auto-filters to assigned + non-completed jobs sorted by most recently updated. Job cards with Start/Resume one-click action. “My Work” nav button in header (first link). Remaining gaps: OPA policy enforcement to restrict browsing, “Next” button to auto-advance to next job.

- [x] Video-level classification (whole-procedure tags, case metadata)
  - Leading Platform: Global classifications and project-level structures; supports bulk classification.
  - CVAT Community gap: No native video-level tag; issue #4952 calls out whole-video/range tagging.
  - Build notes:
    - Add `GlobalClassification` at the task/job level (not frame-level) stored as media metadata.
    - Display separately from timeline interval tags (e.g., procedure type, complication present, video quality).
  - Status: `JobClassification` model (job + label, unique constraint) with migration 0101. REST API (GET/POST/DELETE /jobs/{id}/classifications). `VideoClassificationEditor` component in surgery sidebar "Case info" tab: colored tag chips with close-to-remove, label dropdown filtered to unused labels. Remaining gaps: attribute values per classification, bulk classification across jobs.

## P1 checklist (scale to 20–50 surgeons)

- [ ] Ontology versioning + audit trail for label schema changes
  - Leading Platform: Ontology edit history with user/time versioning and revert support; soft deletion/archive semantics.
  - CVAT Community gap: Labels can be edited, but there’s no first-class ontology version history or migrations.
  - Build notes:
    - Implement a “Procedure Ontology Registry” with immutable versions and migration scripts.
    - Allow projects/tasks to pin to a version; add diff UI and deprecate/alias semantics.
    - Store ontology edits as events for traceability.

- [ ] Dataset-first project model (attach new data units over time)
  - Leading Platform: Projects label data from datasets attached to the project.
  - CVAT Community gap: Tasks are created with fixed data; you can’t add more data later (must create a new task).
  - Build notes:
    - Add a “Dataset” entity (or a high-level abstraction over tasks) to support continuous ingestion while keeping ontology/workflow consistent.
    - Provide “attach dataset” operations and a stable case ID so narration/labels link across derived clips.

- [x] Bulk ingest from cloud storage folders + auto-task creation at scale
  - Leading Platform: Cloud integrations + metadata-driven import; signed URLs can be used without storing them.
  - CVAT Community gap: Cloud storage attachment exists, but multi-task creation can’t upload multiple tasks from cloud storage.
  - Build notes:
    - Build a server-side ingestion service: scan prefixes, validate codecs, generate manifests, create tasks/jobs.
    - Add “procedure templating” (ontology + job settings) and map case → tasks automatically.
  - Status: `bulk_ingest.py` module scans S3 for LeRobot-format episodes (`{prefix}/videos/chunk-XXX/observation.images.endoscope/episode_*.mp4`). Creates one CVAT task per episode with cloud storage attachment, auto-classifies with procedure type, optionally triggers weak labeling. `POST /api/bulk-ingest` API enqueues to RQ IMPORT_DATA queue. `BulkIngestPage` UI at `/bulk-ingest` with cloud storage ID, prefix, project, and weak labeling toggle.

- [x] Webhooks aligned with workflow stages (submitted/reviewed/completed)
  - Leading Platform: Workflow-stage webhooks (e.g., at submit and completion stages).
  - CVAT Community gap: Webhooks are event-based; domain events for surgeon workflow states need to be defined.
  - Build notes:
    - Define a workflow state machine that emits webhook events with stable schemas.
    - Add webhook templates for transcription, SAM3 inference, dataset export, QA sampling, etc.
  - Status: Narration and transcript models wired to webhook dispatch (pre_save/post_save signal handlers + serializer registry). `submitted:procedure` event emitted on job completion with rich surgery context (classifications, interval count, transcript status). `emit_inference_requested()` helper ready for weak labeling pipeline. Event types: `create:narration`, `create:transcript`, `update:transcript`, `submitted:procedure`.

- [ ] Leading Platform-like workflow builder (routers, policy-driven reassignment, reassessment controls)
  - Leading Platform: Routers (percentage/collaborator) + reassessment controls + hide-unassigned semantics.
  - CVAT Community gap: Job stages are fixed (annotation/validation/acceptance) and don’t support routers/multi-branch workflows.
  - Build notes:
    - Implement orchestration above CVAT jobs: configurable stage graph, routing policies, reassignment rules, send-back loops.
    - Map down to CVAT job stage/status while keeping richer workflow state in the fork.

- [x] Interval-aware issue tracking (pin/frame/range issues)
  - Leading Platform: Multiple issue types; supports frame range issues for multi-frame data (with some constraints).
  - CVAT Community gap: Issues exist, but the surgeon workflow needs time-range disputes (phase boundary wrong, narration mismatch at t=…).
  - Build notes:
    - Extend issues to attach to annotation object, frame, time range, narration segment, and ontology item.
    - Add assignment/resolution workflows and expose via API for triage dashboards.
  - Status: Extended Issue model with `end_frame` (nullable), `issue_type` (frame/interval/narration), and `narration` FK. Migration 0104. Serializers updated with validation (interval requires end_frame, narration requires narration FK). ViewSet filterable by issue_type. `SurgeryIssues` component in sidebar Issues tab: create issues by type with frame range inputs, resolve inline, color-coded by type. cvat-core `Issue` class extended with endFrame, issueType, narration properties + serialization.

- [ ] Video quality + label quality metrics tailored to long videos
  - Leading Platform: Video-specific quality/validation metrics and “video health” style views.
  - CVAT Community gap: QA exists (ground truth/honeypots/consensus scores), but not a surgeon-centric long-video triage view.
  - Build notes:
    - Build a “Surgery QA Dashboard”: coverage, unlabeled intervals, phase-change density, segmentation anomalies, disagreement hotspots.
    - Optionally add derived media metrics (FPS, blur/sharpness) for triage.

- [ ] Safer consensus UX for surgeon disagreements
  - Leading Platform: Reviewers compare independent annotator outputs and choose best labels/classifications (bulk or individual).
  - CVAT Community gap: Consensus exists (replica jobs/merge/scores) but is limited to tasks/jobs; merges overwrite parent annotations (irreversible).
  - Build notes:
    - Add non-destructive merges (store results as a candidate layer) and a replica comparison UI (A/B/C).
    - Add project-level consensus management and guardrails (automatic backups before merge).

## P2 checklist (parity polish / enterprise patterns)

- [ ] Enterprise identity patterns (SSO) for hospital environments
  - Leading Platform: SSO via SAML or OIDC.
  - CVAT Community gap: SSO is documented as Enterprise-focused (and social auth is also Enterprise).
  - Build notes:
    - If required, implement OIDC/SAML in the fork, or front CVAT with an auth proxy that enforces SSO and maps identities to CVAT users/roles.

- [ ] “Strict client-only access” style data access model
  - Leading Platform: Strict client-only access options + signed URL handling; can avoid platform pre-processing/download.
  - CVAT Community gap: Cloud storage/remote URLs exist, but CVAT typically manages access/chunking for video delivery.
  - Build notes:
    - If required, redesign media delivery so the browser fetches from object storage via short-lived URLs and the backend stores only annotation JSON + metadata.
    - Note: This impacts decoding/seek and is non-trivial.

- [ ] Turnkey analytics deployment & operation
  - Leading Platform: Dashboards/analytics as part of the platform experience.
  - CVAT Community gap: Analytics stack is deploy/setup heavy and is more platform monitoring than surgeon-centric ops dashboards.
  - Build notes:
    - Package analytics as a default-enabled module in the fork.
    - Add surgeon/program dashboards (time spent, throughput) and keep ops dashboards for admins.

- [ ] Developer ergonomics for large customization (module boundaries & where to patch)
  - Leading Platform: Cohesive product surface; extend via SDK and integrations.
  - CVAT Community gap: Repo modularity exists, but domain logic can easily sprawl without guardrails.
  - Build notes:
    - Split workstreams by module: UI (`cvat-ui/`), timeline/decoding (`cvat-data/`), APIs/SDK (`cvat-sdk/`), inference hooks (`serverless/`).
    - Add a new “surgery” module to avoid spraying domain logic across core.

Build blueprint for a CVAT fork that supports surgeon narration and temporal labeling
This section is the “how to build it” scaffolding for an agentic SWE team. It uses CVAT’s own documented system/extension points so the work stays maintainable.

Leverage CVAT’s existing extensibility rather than rewriting the platform
CVAT already provides:

A webhook system (GUI + API) designed for event-triggered callbacks.
A REST API for all clients and an API schema endpoint (/api/docs).
A serverless / automatic annotation path where CVAT can run models for AI-assisted labeling (including via serverless deployment patterns).
A modular repo structure separating UI, decoding, SDK, and serverless models.
Policy control via OPA .rego files for self-hosted role adjustments.
Your fork should preserve these primitives and add surgical-domain concepts by augmenting the data model and UI, not by replacing the core.

Add three first-class data concepts that CVAT does not model today
Leading Platform’s “datasets attached to projects” model implies a persistent dataset entity separate from annotation units.
 CVAT tasks, by contrast, are created with fixed data (cannot add data later).

For surgical workflows, introduce:

Case: a single procedure recording (or original source).
Clip/Segment: derived time-span slices (e.g., phases or regions of interest).
Procedure Template: ontology + labeling instructions + required tracks (segmentations, phases, gestures) + QA rules.
These can be represented as additional tables/entities in your fork that reference CVAT’s existing Project → Task → Job hierarchy (CVAT’s hierarchy is explicitly Project → Task → Job).

Implement temporal labeling as a native primitive (not a post-process hack)
A core reason surgeons like Leading Platform for video labeling is that “range” labeling is an explicit UI affordance: start/end/range controls.
 CVAT’s current shortcomings here are acknowledged by CVAT itself through the open request for video-level/range tagging.

Treat temporal labeling as first-class:

Interval record (start_frame/end_frame) + label + attributes.
Timeline editor that supports: snapping, splitting, merging, overlap rules, and “phase track” constraints (e.g., phases shouldn’t overlap; gestures may overlap).
Export that yields intervals directly (rather than forcing per-frame duplication).
Implement narration as an artifact with strong provenance signals
Leading Platform’s audio tooling provides the conceptual basis: audio regions are defined in millisecond ranges; audio metadata can be provided for alignment.
 For narration you need two extras:

Capture audio inside the labeling session, aligned to the video timeline.
Produce artifacts suitable for downstream reasoning dataset distillation (audio, transcript, and time-aligned segments).
Your fork should store:

raw audio (or chunked audio)
transcript versions (draft/final)
alignment metadata (frame ↔ time mapping; offsets; playback speed adjustments)
audit fields (user, timestamps)
Then emit webhook events so downstream pipelines can run without polling. CVAT already has a webhook framework; you add new event types.

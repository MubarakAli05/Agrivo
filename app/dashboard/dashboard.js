"use strict";

const byId = (id) => document.getElementById(id);
const display = (value) => value === null || value === undefined ? "Not reported" :
  typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
const put = (id, value) => { byId(id).textContent = display(value); };
function card(parent, title, value, note = "") {
  const item = document.createElement("article");
  item.className = "card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const content = document.createElement("p");
  content.className = "value";
  content.textContent = display(value);
  item.append(heading, content);
  if (note) {
    const detail = document.createElement("p");
    detail.textContent = note;
    item.append(detail);
  }
  parent.append(item);
}
async function request(path, options = {}) {
  const response = await fetch(path, { ...options, credentials: "omit", cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || "Local service unavailable");
  return payload;
}
function renderStatus(payload) {
  const workspace = payload.workspace;
  const dataset = workspace.synthetic_dataset || {};
  const sourceReport = payload.sources;
  const training = display(workspace.training);
  put("model-badge", /not started|zero optimizer|no neural training/i.test(training)
    ? "UNTRAINED NEURAL MODEL" : "MODEL TRAINING: SEE VALIDATED STATUS");
  const cards = byId("status-cards");
  cards.replaceChildren();
  card(cards, "System / API", `${workspace.status || "UNKNOWN"} / ${payload.api.status}`, "GREEN = pass · YELLOW = review · RED = blocked");
  card(cards, "Data scope", workspace.data_used, "Synthetic demonstrations are not agricultural field observations.");
  card(cards, "Model", workspace.model, "UNTRAINED unless a validated training report explicitly confirms otherwise.");
  card(cards, "Training", workspace.training, "Read-only reporting. Training cannot be started here.");
  put("next-step", workspace.next_required_step);
  const blockers = byId("blockers");
  blockers.replaceChildren();
  for (const blocker of workspace.blockers || []) {
    const item = document.createElement("li");
    item.textContent = display(blocker);
    blockers.append(item);
  }
  const metrics = byId("metrics");
  metrics.replaceChildren();
  card(metrics, "Discovered datasets", sourceReport?.source_count, "Registry metadata; not downloaded datasets.");
  card(metrics, "Downloaded datasets / images indexed", workspace.downloaded_datasets ?? null,
    `Images indexed: ${display(workspace.retrieval?.images_indexed)}. No count is inferred from metadata.`);
  card(metrics, "Synthetic fixture counts", dataset.counts, "Soil records, plant metadata and QA examples; not indexed field data.");
  card(metrics, "Soil records indexed", workspace.retrieval?.soil_records_indexed);
  card(metrics, "Missing soil values / duplicate records", dataset.missing_soil_values,
    `Duplicate fixture records: ${display(dataset.duplicate_records)}. Missing external metadata: not reported.`);
  card(metrics, "Tokenizer vocabulary", workspace.tokenizer?.text_vocab_size,
    `Structured vocabulary: ${display(workspace.tokenizer?.structured_vocab_size)} tokens.`);
  card(metrics, "Transformer parameters / checkpoints", workspace.model_check?.checks?.parameters,
    `Checkpoint status: ${display(workspace.checkpoint ?? workspace.inference?.checkpoint)}. Architecture checks are not trained weights.`);
  card(metrics, "Training / validation loss / perplexity", workspace.metrics, "Mechanics checks are not trained model quality.");
  card(metrics, "Retrieval successes / failures / source usage", workspace.retrieval?.metrics, "Not inferred from answered-question counts.");
  card(metrics, "QA answered / unknown / errors", payload.session, "Only this server session; query counters are not persisted.");
  card(metrics, "Unsupported / hallucination tests", workspace.evaluation ?? dataset.model_hallucination_test, "Not reported does not mean passed.");
  const sources = byId("sources");
  sources.replaceChildren();
  card(sources, "Synthetic release provenance", dataset.data_used ?? "No validated synthetic release reported",
    `SYNTHETIC · ${display(dataset.release_id ?? dataset.dataset_version)}`);
  if (!sourceReport) card(sources, "Source registry unavailable", payload.sources_error || "UNKNOWN");
  for (const [id, source] of Object.entries(sourceReport?.sources || {})) {
    card(sources, source.name || id, `${source.license_gate || "UNKNOWN"} · ${source.url || "URL not reported"}`,
      `Metadata only. Provenance / license / approval: ${display({ version: source.version ?? source.dataset_version, license: source.license, ingestion: source.ingestion })}`);
  }
  put("raw-status", payload);
  put("status-message", `Local status loaded at ${new Date().toLocaleTimeString()}. Phase ${workspace.phase ?? "UNKNOWN"}; ${workspace.scope || "scope not reported"}.`);
}
async function refresh() {
  byId("refresh").disabled = true;
  put("status-message", "Reading local workspace…");
  try { renderStatus(await request("/api/status")); }
  catch (error) {
    put("status-message", `UNKNOWN: ${error.message}. Any previously displayed values may be stale.`);
  } finally { byId("refresh").disabled = false; }
}
byId("refresh").addEventListener("click", refresh);
byId("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (!question) { put("query-state", "Enter a nonempty question."); return; }
  byId("ask").disabled = true;
  byId("answer-panel").hidden = true;
  put("query-state", "Querying local evidence… no external request is made.");
  try {
    const answer = await request("/api/answer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    const unknown = answer.unknown !== false;
    put("answer-badge", unknown ? "UNKNOWN / CHECK EVIDENCE" : "LOCAL RESPONSE / CHECK PROVENANCE");
    put("answer-text", answer.answer ?? answer.response ?? "UNKNOWN: backend returned no answer text");
    put("evidence", { evidence: answer.evidence ?? answer.context ?? answer.retrieval ?? null,
      citations: answer.citations ?? answer.sources ?? answer.provenance ?? null });
    put("answer-details", answer);
    put("query-state", "Local response received. Inspect provenance and limitations before use.");
  } catch (error) {
    put("answer-badge", "UNKNOWN");
    put("answer-text", `UNKNOWN: ${error.message}`);
    put("evidence", "No verified evidence returned.");
    put("answer-details", "Check local data, checkpoint and source approvals. This page will not create them.");
    put("query-state", "Query could not be completed.");
  } finally {
    byId("answer-panel").hidden = false;
    byId("ask").disabled = false;
  }
});
refresh();

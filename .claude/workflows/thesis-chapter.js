export const meta = {
  name: 'thesis-chapter',
  description: 'Draft thesis chapters through an evidence-first 4-agent pipeline with an adversarial audit',
  whenToUse: 'When drafting or redrafting one or more thesis chapters from the repo evidence and literature',
  phases: [
    { title: 'Evidence', detail: 'extract verified facts with source locations' },
    { title: 'Literature', detail: 'find and verify citations for the chapter' },
    { title: 'Draft', detail: 'write from supplied evidence only' },
    { title: 'Audit', detail: 'adversarial check for fabrications and overstatement' },
    { title: 'Revise', detail: 'apply audit findings' },
  ],
}

// args: { chapters: [{ id, title, brief, evidence, literature }], outDir, revise }
// Falls back to a single demo chapter if invoked with no args.
const cfg = args || {}
const OUT = cfg.outDir || 'Thesis_report/chapters'
const REVISE = cfg.revise !== false

const CHAPTERS = cfg.chapters && cfg.chapters.length ? cfg.chapters : [{
  id: 'ch7-active-learning',
  title: 'Active Learning',
  brief: 'The 8-stage AL cycle, residual-GP acquisition, and what the exp059 cycle actually showed.',
  evidence: 'The AL pipeline stages and their code; the acquisition modes; the exp059 gp_error run '
          + 'results including every decision-ledger claim and its verdict and supporting numbers; '
          + 'the ECAD budget per cycle; pre/post fine-tune p99 MAE.',
  literature: 'Active learning and acquisition-function work relevant to PI surrogates; residual/error '
            + 'surrogates; GP-UCB; anything competing with GP-based acquisition for simulation budget.',
}]

const FACTS_SCHEMA = {
  type: 'object',
  required: ['facts', 'missing'],
  properties: {
    topic: { type: 'string' },
    facts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim', 'source', 'authority'],
        properties: {
          claim: { type: 'string' },
          value: { type: 'string' },
          source: { type: 'string' },
          authority: { type: 'string', enum: ['run-artifact', 'config', 'source-code', 'vault-note'] },
          tension: { type: 'boolean' },
          note: { type: 'string' },
        },
      },
    },
    missing: { type: 'array', items: { type: 'string' } },
    contradictions: { type: 'array', items: { type: 'string' } },
  },
}

const LIT_SCHEMA = {
  type: 'object',
  required: ['papers', 'gaps'],
  properties: {
    query: { type: 'string' },
    papers: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'authors', 'year', 'url', 'verified', 'relevance'],
        properties: {
          title: { type: 'string' },
          authors: { type: 'string' },
          year: { type: 'number' },
          venue: { type: 'string' },
          url: { type: 'string' },
          verified: { type: 'string', enum: ['yes', 'partial'] },
          unconfirmed: { type: 'string' },
          relevance: { type: 'string' },
          competes_with_thesis: { type: 'boolean' },
        },
      },
    },
    known: { type: 'array', items: { type: 'string' } },
    gaps: { type: 'array', items: { type: 'string' } },
  },
}

const AUDIT_SCHEMA = {
  type: 'object',
  required: ['verdict', 'findings'],
  properties: {
    verdict: { type: 'string', enum: ['PASS', 'NEEDS_WORK', 'REJECT'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'location', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['fabrication', 'unsupported', 'overstated', 'inconsistent', 'minor'] },
          location: { type: 'string' },
          problem: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
    unresolved_todos: { type: 'array', items: { type: 'string' } },
    verified_clean: { type: 'array', items: { type: 'string' } },
  },
}

log(`Drafting ${CHAPTERS.length} chapter(s) into ${OUT}/`)

// Each chapter runs its own full pipeline. No barrier between chapters: a chapter that
// finishes evidence early starts drafting while another is still gathering.
const results = await pipeline(
  CHAPTERS,

  // 1+2 — evidence and literature run concurrently; the draft needs both.
  async (ch) => {
    const [facts, lit] = await parallel([
      () => agent(
        `Extract verified facts for thesis chapter "${ch.title}".\n\n`
        + `Chapter brief: ${ch.brief}\n\nNeeded specifically:\n${ch.evidence}\n\n`
        + `Read the repository and thesis_vault directly. Every fact needs a file location. `
        + `Anything you cannot locate goes in "missing".`,
        { agentType: 'evidence-extractor', schema: FACTS_SCHEMA, phase: 'Evidence',
          effort: 'medium', label: `facts:${ch.id}` },
      ),
      () => agent(
        `Find related work for thesis chapter "${ch.title}".\n\n`
        + `Chapter brief: ${ch.brief}\n\nFocus:\n${ch.literature}\n\n`
        + `Power integrity only.\n\n`
        + `EFFICIENCY: 40 papers are ALREADY curated and verified under thesis_vault/05-Literature/, `
        + `with a per-chapter mapping in "Literature Review.md" and bib keys in Thesis_report/refs.bib. `
        + `Read that mapping first. Anything already covered there goes straight into "known" WITHOUT `
        + `re-fetching or re-verifying it — that work is done. Run web searches ONLY for aspects the `
        + `vault genuinely does not cover, and fetch source pages only for those new papers.`,
        { agentType: 'literature-researcher', schema: LIT_SCHEMA, phase: 'Literature',
          effort: 'medium', label: `lit:${ch.id}` },
      ),
    ])
    return { ch, facts, lit }
  },

  // 3 — draft from the two blocks only. The writer has no repo or web access.
  async (prev) => {
    if (!prev || !prev.facts) return null
    const { ch, facts, lit } = prev
    const path = `${OUT}/${ch.id}.tex`
    const summary = await agent(
      `Draft thesis chapter "${ch.title}" and write it to ${path}\n\n`
      + `Chapter brief: ${ch.brief}\n\n`
      + `FACTS (the only facts you may assert):\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE (the only citations you may use):\n${JSON.stringify(lit && lit.papers || [], null, 1)}\n\n`
      + `Target ~60% coverage. Anything not in those blocks becomes \\fillin{...} or a one-line \\figplace{...}. Do not fill gaps, do not build figure environments.`,
      { agentType: 'thesis-writer', phase: 'Draft', effort: 'high', label: `draft:${ch.id}` },
    )
    return { ch, facts, lit, path, summary }
  },

  // 4 — adversarial audit against the same two blocks.
  async (prev) => {
    if (!prev) return null
    const { ch, facts, lit, path } = prev
    const audit = await agent(
      `Audit the drafted chapter at ${path} against the evidence it was built from.\n\n`
      + `FACTS:\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE:\n${JSON.stringify(lit && lit.papers || [], null, 1)}\n\n`
      + `Assume the draft is wrong until each claim is traced. Verify numbers against the artifacts `
      + `yourself — do not trust the draft or the FACTS block blindly.`,
      { agentType: 'consistency-reviewer', schema: AUDIT_SCHEMA, phase: 'Audit',
        effort: 'high', label: `audit:${ch.id}` },
    )
    return { ...prev, audit }
  },

  // 5 — one corrective pass, only when the audit found something actionable.
  async (prev) => {
    if (!prev || !prev.audit) return prev
    const { ch, facts, lit, path, audit } = prev
    const actionable = (audit.findings || []).filter(
      (f) => f.severity !== 'minor',
    )
    if (audit.verdict === 'PASS' || !actionable.length) {
      log(`${ch.id}: ${audit.verdict} — no revision needed`)
      return prev
    }
    log(`${ch.id}: ${audit.verdict}, ${actionable.length} finding(s) — revising`)
    await agent(
      `Revise the chapter at ${path} to fix these audit findings.\n\n`
      + `${JSON.stringify(actionable, null, 1)}\n\n`
      + `FACTS:\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE:\n${JSON.stringify(lit && lit.papers || [], null, 1)}\n\n`
      + `Any claim you cannot support from those blocks must become a \\fillin{...} marker `
      + `or be deleted. Never satisfy a finding by inventing support for the claim.`,
      { agentType: 'thesis-writer', phase: 'Revise', effort: 'high', label: `revise:${ch.id}` },
    )
    return { ...prev, revised: true }
  },
)

const done = results.filter(Boolean)
return {
  chapters: done.map((r) => ({
    id: r.ch.id,
    path: r.path,
    verdict: r.audit && r.audit.verdict,
    findings: (r.audit && r.audit.findings || []).length,
    fabrications: (r.audit && r.audit.findings || []).filter((f) => f.severity === 'fabrication').length,
    open_todos: (r.audit && r.audit.unresolved_todos || []).length,
    evidence_gaps: (r.facts && r.facts.missing || []).length,
    revised: !!r.revised,
  })),
  note: 'Open TODO markers and evidence gaps are real holes — resolve them before submission.',
}

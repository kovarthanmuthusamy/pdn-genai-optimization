export const meta = {
  name: 'thesis-phase1',
  description: 'Draft only the experiment-independent chapters: theory, background, related work, design and implementation',
  whenToUse: 'Before the remaining experiments finish. Writes the stable half of the thesis so that afterwards only results and analysis remain.',
  phases: [
    { title: 'Evidence', detail: 'extract verified facts with source locations' },
    { title: 'Literature', detail: 'match chapters to the already-verified vault papers' },
    { title: 'Draft', detail: 'write LaTeX from supplied evidence only' },
    { title: 'Audit', detail: 'adversarial check for fabrications and overstatement' },
    { title: 'Revise', detail: 'apply audit findings' },
  ],
}

// PHASE 1 = everything that does NOT depend on experiments still to be run.
// Deliberately excluded and left for phase 2:
//   ch4 Results          - needs Stage 2 on exp059, ablations, exp060
//   ch5 Conclusion       - needs results to answer the research questions
//   appendix C           - supplementary results
//   ch1 sec:objectives_contributions - needs supervisor-agreed research questions
const PHASE1 = [
  {
    id: 'ch1-introduction',
    title: 'Introduction',
    brief: 'Motivation and problem statement, and the structure of the thesis. Write '
         + 'sec:motivation_problem and sec:thesis_structure fully. For '
         + 'sec:objectives_contributions write only what the repository demonstrably supports and '
         + 'leave the research questions and claimed contributions as \\fillin - they need '
         + 'supervisor agreement and must not be invented.',
    evidence: 'The engineering problem: target impedance mask, 1-600 MHz over 231 bins, binary '
            + 'placement over 52 slots, budget K, the 2^52 search space, the conventional '
            + 'place-simulate-inspect-adjust loop and why it is expensive. The two-stage framework '
            + 'and the analysis layer. The approved chapter structure in Thesis_report/thesis_v3.tex. '
            + 'Do NOT source anything from the root README.md - it is stale.',
    literature: 'Target impedance methodology; the cost of conventional PDN design iteration; '
              + 'learned approaches to decap placement as context for the problem.',
  },
  {
    id: 'ch2-background',
    title: 'Theoretical Background',
    brief: 'PDN impedance behaviour and decoupling, PI simulation and multi-modal design data, the '
         + 'formal decap placement problem; then VAEs and multi-modal learning, physics-informed '
         + 'and surrogate modelling, latent-space optimization; then related work and the research '
         + 'gap. This chapter is fully independent of the pending experiments - draft it completely.',
    evidence: 'The modality definitions and tensor shapes (occupancy 52, spectrum 231, heatmap '
            + '64x64), the frequency anchors, the formal feasibility condition and per-K objective '
            + 'with the safety margin, and the PoE / structured-latent / GNN / FiLM design as '
            + 'actually implemented in exp059. Distinguish what the current model does from what '
            + 'earlier experiments did.',
    literature: 'The full related-work sweep. CRITICAL: Prof. Jürgen Götze, the examiner, is a '
              + 'co-author of arXiv:2606.07463 (Amortized Neural Optimization with differentiable '
              + 'surrogates), arXiv:2605.18170 and arXiv:2606.15234. The research gap MUST engage '
              + 'with that work explicitly. Also position against DevFormer, ConvGA, the KAIST RL '
              + 'line, the PDN impedance surrogates, and the IR-drop spatial map literature.',
  },
  {
    id: 'ch3-design',
    title: 'Design and Implementation',
    brief: 'Data preparation, the multi-modal VAE surrogate, latent-space optimization, the '
         + 'experimental setup, and the active-learning method. Describe the METHOD and what is '
         + 'IMPLEMENTED - not results. Where a design rationale was never written down, leave '
         + '\\fillin rather than reconstructing intent.',
    evidence: 'Dataset construction and robust normalization; splitting, stratification and '
            + 'layout-level holdout; the exp059 architecture and its config values; loss terms and '
            + 'physics regularization; the training loop, schedules and DDP; the latent '
            + 'optimization formulation, the straight-through top-K read-out and the selection '
            + 'rule; the 8-stage AL cycle and residual-GP acquisition. Two things to flag rather '
            + 'than paper over: (1) the ToC term "Auxiliary Guidance Network" maps to no code of '
            + 'that name - the closest is surrogate_impedance.py, which exists only in exp037/038/040; '
            + '(2) pipelines/latent/optimize.py still defaults to exp038_true_multi, so the '
            + 'implemented Stage 2 is not yet wired to exp059.',
    literature: 'Method-level citations only: PoE multimodal VAE, conditional VAE, FiLM, GCN and '
              + 'message passing, the straight-through estimator, latent-space inverse design, '
              + 'GP-UCB and residual/error surrogates. Most are already in the vault.',
  },
  {
    id: 'appendix-a-simulation-dataset',
    title: 'Appendix A: Power Integrity Simulation Environment and Dataset Specification',
    brief: 'The simulation environment and the dataset specification as they exist. Reference '
         + 'material, tables over prose.',
    evidence: 'The ECADSTAR PI simulation setup and how layouts are exported and simulated; the '
            + 'dataset directory layout, dataset_meta.json fields, normalization statistics, '
            + 'frequency anchors and sample counts. State clearly that datasets are untracked and '
            + 'give the paths.',
    literature: 'None expected - this is project reference material.',
  },
  {
    id: 'appendix-b-hyperparameters',
    title: 'Appendix B: Hyperparameter Configuration for Training and Optimisation',
    brief: 'Hyperparameter tables for the current model and the optimizer. Almost entirely tabular; '
         + 'keep prose to a short orienting paragraph per table.',
    evidence: 'experiments/exp059_capacity_freq/config.yaml (183 keys) grouped into architecture, '
            + 'training schedule, loss weights and AL fine-tune overrides; the config variants; and '
            + 'the CONFIG block constants in pipelines/latent/optimize.py. Values verbatim.',
    literature: 'None expected.',
  },
]

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
          title: { type: 'string' }, authors: { type: 'string' }, year: { type: 'number' },
          venue: { type: 'string' }, url: { type: 'string' },
          verified: { type: 'string', enum: ['yes', 'partial'] },
          unconfirmed: { type: 'string' }, relevance: { type: 'string' },
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
          location: { type: 'string' }, problem: { type: 'string' },
          evidence: { type: 'string' }, fix: { type: 'string' },
        },
      },
    },
    unresolved_todos: { type: 'array', items: { type: 'string' } },
    verified_clean: { type: 'array', items: { type: 'string' } },
  },
}

const OUT = (args && args.outDir) || 'Thesis_report/chapters'
const SELECTED = (args && args.only && args.only.length)
  ? PHASE1.filter((c) => args.only.includes(c.id))
  : PHASE1

log(`Phase 1: ${SELECTED.length} experiment-independent chapter(s) -> ${OUT}/`)
log('Excluded by design: ch4 Results, ch5 Conclusion, appendix C (all need pending experiments)')

const results = await pipeline(
  SELECTED,

  async (ch) => {
    const [facts, lit] = await parallel([
      () => agent(
        `Extract verified facts for thesis chapter "${ch.title}".\n\n`
        + `Chapter brief: ${ch.brief}\n\nNeeded specifically:\n${ch.evidence}\n\n`
        + `This chapter is being written BEFORE the remaining experiments run. Report method, `
        + `architecture, configuration and implemented behaviour. Do NOT report or imply results `
        + `from runs that have not happened. Anything you cannot locate goes in "missing".`,
        { agentType: 'evidence-extractor', schema: FACTS_SCHEMA, phase: 'Evidence',
          effort: 'medium', label: `facts:${ch.id}` },
      ),
      () => agent(
        `Find related work for thesis chapter "${ch.title}".\n\n`
        + `Chapter brief: ${ch.brief}\n\nFocus:\n${ch.literature}\n\n`
        + `Power integrity only.\n\n`
        + `EFFICIENCY: 40 papers are ALREADY curated and verified under thesis_vault/05-Literature/, `
        + `with a per-chapter mapping in "Literature Review.md" and bib keys in Thesis_report/refs.bib. `
        + `Read that mapping first. Anything already covered goes straight into "known" WITHOUT `
        + `re-fetching. Search the web ONLY for aspects the vault genuinely does not cover.`,
        { agentType: 'literature-researcher', schema: LIT_SCHEMA, phase: 'Literature',
          effort: 'medium', label: `lit:${ch.id}` },
      ),
    ])
    return { ch, facts, lit }
  },

  async (prev) => {
    if (!prev || !prev.facts) return null
    const { ch, facts, lit } = prev
    const path = `${OUT}/${ch.id}.tex`
    const summary = await agent(
      `Draft thesis chapter "${ch.title}" and write it to ${path}\n\n`
      + `Chapter brief: ${ch.brief}\n\n`
      + `FACTS (the only facts you may assert):\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE (the only citations you may use):\n${JSON.stringify((lit && lit.papers) || [], null, 1)}\n\n`
      + `This is a method/theory chapter written before the remaining experiments. Never state or `
      + `imply an experimental outcome. Target ~60% coverage; unsure counts as missing. Gaps become `
      + `\\fillin{...} or a one-line \\figplace{...}.`,
      { agentType: 'thesis-writer', phase: 'Draft', effort: 'high', label: `draft:${ch.id}` },
    )
    return { ch, facts, lit, path, summary }
  },

  async (prev) => {
    if (!prev) return null
    const { ch, facts, lit, path } = prev
    const audit = await agent(
      `Audit the drafted chapter at ${path} against the evidence it was built from.\n\n`
      + `FACTS:\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE:\n${JSON.stringify((lit && lit.papers) || [], null, 1)}\n\n`
      + `Additional check for this phase: the chapter must not state or imply any experimental `
      + `result. Stage 2 has never been run on exp059, there are no ablation artifacts, and exp060 `
      + `has empty metrics and checkpoints. Flag any sentence that reads as a finding.\n\n`
      + `Assume the draft is wrong until each claim is traced. Verify numbers yourself.`,
      { agentType: 'consistency-reviewer', schema: AUDIT_SCHEMA, phase: 'Audit',
        effort: 'high', label: `audit:${ch.id}` },
    )
    return { ...prev, audit }
  },

  async (prev) => {
    if (!prev || !prev.audit) return prev
    const { ch, facts, lit, path, audit } = prev
    const actionable = (audit.findings || []).filter((f) => f.severity !== 'minor')
    if (audit.verdict === 'PASS' || !actionable.length) {
      log(`${ch.id}: ${audit.verdict} - no revision needed`)
      return prev
    }
    log(`${ch.id}: ${audit.verdict}, ${actionable.length} finding(s) - revising`)
    await agent(
      `Revise the chapter at ${path} to fix these audit findings.\n\n`
      + `${JSON.stringify(actionable, null, 1)}\n\n`
      + `FACTS:\n${JSON.stringify(facts, null, 1)}\n\n`
      + `LITERATURE:\n${JSON.stringify((lit && lit.papers) || [], null, 1)}\n\n`
      + `Any claim you cannot support becomes \\fillin{...} or is deleted. Never satisfy a finding `
      + `by inventing support for the claim.`,
      { agentType: 'thesis-writer', phase: 'Revise', effort: 'high', label: `revise:${ch.id}` },
    )
    return { ...prev, revised: true }
  },
)

const done = results.filter(Boolean)
return {
  phase: 1,
  chapters: done.map((r) => ({
    id: r.ch.id,
    path: r.path,
    verdict: r.audit && r.audit.verdict,
    fabrications: ((r.audit && r.audit.findings) || []).filter((f) => f.severity === 'fabrication').length,
    open_fillins: ((r.audit && r.audit.unresolved_todos) || []).length,
    evidence_gaps: ((r.facts && r.facts.missing) || []).length,
    revised: !!r.revised,
  })),
  deferred_to_phase2: ['ch4-results', 'ch5-conclusion', 'appendix-c-supplementary',
                       'ch1 sec:objectives_contributions (needs supervisor-agreed research questions)'],
}

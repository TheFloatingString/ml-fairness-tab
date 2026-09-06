import { promises as fs } from "fs";
import path from "path";

// The experiment writes its JSON artifacts to <repo>/results; the Next.js app
// lives in <repo>/web, so climb one level out of process.cwd().
export const RESULTS_DIR = path.join(process.cwd(), "..", "results");

const DEFAULT_DATASET = "adult";

interface GapBlock {
  by_group?: Record<string, number>;
  by_group_se?: Record<string, number>;
  diff: number;
  abs_diff?: number;
  diff_se?: number;
}

export interface AttrMetrics {
  accuracy: number;
  subgroup_accuracy: Record<string, number>;
  subgroup_count: Record<string, number>;
  accuracy_diff: number;
  accuracy_diff_se?: number;
  demographic_parity: GapBlock;
  equal_opportunity: GapBlock;
  equalized_odds: GapBlock & {
    by_group_tpr: Record<string, number>;
    by_group_fpr: Record<string, number>;
  };
}

export interface SweepRow {
  model: string;
  gpu: string;
  dtype: string;
  device_name: string;
  accuracy: number;
  accuracy_diff: number | null;
  accuracy_diff_se?: number | null;
  demographic_parity_diff: number;
  demographic_parity_diff_se?: number | null;
  equal_opportunity_diff?: number;
  equal_opportunity_diff_se?: number | null;
  equalized_odds_diff: number;
  equalized_odds_diff_se?: number | null;
  primary_attribute?: string;
  by_attribute?: Record<string, AttrMetrics>;
  subgroup_accuracy: Record<string, number>;
  subgroup_selection_rate: Record<string, number>;
  subgroup_tpr: Record<string, number>;
  subgroup_fpr: Record<string, number>;
  subgroup_count: Record<string, number>;
}

export interface DriftPoint {
  gpu: string;
  dataset: string;
  x: number;
  y: number;
}

export interface DriftCorrelation {
  model: string;
  reference_gpu: string;
  n_points: number;
  // null when Pearson R is undefined (e.g. near-constant drift -> NaN, which
  // parseLenient maps to null).
  pearson_r: number | null;
  p_value: number | null;
  points: DriftPoint[];
}

export interface CorrectionEntry {
  acc_weight: number;
  correction_coeff: number | number[];
  per_gpu_accuracy: Record<string, number>;
  per_gpu_demographic_parity_diff?: Record<string, number>;
  accuracy_vs_t4?: Record<string, number>;
  demographic_parity_diff_vs_t4?: Record<string, number>;
  mean_accuracy: number;
  accuracy_spread: number;
  demographic_parity_diff_spread: number;
}

export interface CorrectionResult {
  model: string;
  dataset: string;
  dtype: string;
  gpus: string[];
  reference_gpu?: string;
  n_calib: number;
  n_test: number;
  mean_l2_logit_drift_vs_t4?: Record<string, number>;
  baseline: Omit<CorrectionEntry, "acc_weight" | "correction_coeff">;
  corrected: CorrectionEntry[];
}

/** E-value analysis (results/evalues/{logit,metric}_level.json). Overwhelming
 *  evidence is serialised as the string "inf" by evalue_analysis.py. */
export type Ev = number | "inf";

export interface EvalueCombinedRow {
  model: string;
  gpu: string;
  dtype?: string;
  attribute?: string;
  test: string;
  evalue_product: Ev;
  "ebh_reject@0.05": boolean;
  "ebh_reject@0.1": boolean;
}

export interface EvaluePerDatasetRow {
  dataset: string;
  model: string;
  gpu: string;
  dtype?: string;
  attribute?: string;
  test: string;
  evalue: Ev;
  // logit-level extras
  mean_abs_logodds_drift?: number;
  max_abs_logodds_drift?: number;
  argmax_flips_vs_ref?: number;
  deterministic?: boolean;
  noise_kind?: string;
}

export interface EvalueLevel {
  alpha: number;
  n_hypotheses: number;
  combined: EvalueCombinedRow[];
  per_dataset: EvaluePerDatasetRow[];
}

export interface Evalues {
  logit: EvalueLevel | null;
  metric: EvalueLevel | null;
}

/** Python's json.dump emits bare NaN/Infinity, which JSON.parse rejects. */
function parseLenient<T>(text: string): T {
  return JSON.parse(
    text.replace(/\b-?Infinity\b/g, "null").replace(/\bNaN\b/g, "null"),
  ) as T;
}

async function readJson<T>(file: string): Promise<T> {
  return parseLenient<T>(await fs.readFile(path.join(RESULTS_DIR, file), "utf8"));
}

async function listResultFiles(prefix: string): Promise<string[]> {
  let entries: string[];
  try {
    entries = await fs.readdir(RESULTS_DIR);
  } catch {
    return [];
  }
  return entries.filter((f) => f.startsWith(prefix) && f.endsWith(".json"));
}

/** "results.json" -> "adult"; "results-german_credit.json" -> "german_credit". */
function datasetFromFilename(file: string, stem: string): string {
  const m = file.match(new RegExp(`^${stem}-(.+)\\.json$`));
  return m ? m[1] : DEFAULT_DATASET;
}

export async function loadSweep(): Promise<Record<string, SweepRow[]>> {
  const files = await listResultFiles("results");
  const out: Record<string, SweepRow[]> = {};
  for (const file of files) {
    out[datasetFromFilename(file, "results")] = await readJson<SweepRow[]>(file);
  }
  return out;
}

/** correction_results-<dataset>-<model>-<dtype>.json (dataset may contain
 *  underscores; model and dtype never contain a dash). Older files without the
 *  -<dtype> suffix are read as fp32. */
export async function loadCorrection(): Promise<Record<string, CorrectionResult[]>> {
  const files = await listResultFiles("correction_results");
  const out: Record<string, CorrectionResult[]> = {};
  for (const file of files) {
    const withDtype = file.match(
      /^correction_results-(.+)-([^-]+)-(fp32|fp16|bf16)\.json$/,
    );
    const legacy = file.match(/^correction_results-(.+)-([^-]+)\.json$/);
    const m = withDtype ?? legacy;
    if (!m) continue;
    const [, dataset, model] = m;
    const dtype = withDtype ? withDtype[3] : "fp32";
    const res = await readJson<CorrectionResult>(file);
    // Filename is the source of truth for identity (older files omit these).
    res.dataset = dataset;
    res.model = model;
    res.dtype = dtype;
    (out[dataset] ??= []).push(res);
  }
  const dtypeRank = (d: string) => ["fp32", "fp16", "bf16"].indexOf(d);
  for (const list of Object.values(out))
    list.sort(
      (a, b) =>
        a.model.localeCompare(b.model) || dtypeRank(a.dtype) - dtypeRank(b.dtype),
    );
  return out;
}

/** results/evalues/{logit,metric}_level.json (written by evalue_analysis.py).
 *  Absent files are returned as null so the dashboard can show a hint. */
export async function loadEvalues(): Promise<Evalues> {
  const read = async (f: string): Promise<EvalueLevel | null> => {
    try {
      return await readJson<EvalueLevel>(path.join("evalues", f));
    } catch {
      return null;
    }
  };
  return {
    logit: await read("logit_level.json"),
    metric: await read("metric_level.json"),
  };
}

/** drift_correlation-<model>.json, one per in-context model. */
export async function loadDriftCorrelation(): Promise<DriftCorrelation[]> {
  const files = await listResultFiles("drift_correlation");
  const out: DriftCorrelation[] = [];
  for (const file of files) {
    const m = file.match(/^drift_correlation-(.+)\.json$/);
    if (!m) continue;
    const res = await readJson<DriftCorrelation>(file);
    res.model = m[1];
    out.push(res);
  }
  return out.sort((a, b) => a.model.localeCompare(b.model));
}

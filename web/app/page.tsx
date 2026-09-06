"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type {
  CorrectionResult,
  DriftCorrelation,
  Ev,
  EvalueLevel,
  Evalues,
  SweepRow,
} from "@/lib/results";

const GPU_ORDER = ["T4", "L4", "A100", "H100", "B200"];
const DTYPE_ORDER = ["fp32", "fp16", "bf16"];
const MODEL_COLORS: Record<string, string> = {
  tabpfn: "#6ea8fe",
  tabicl: "#f6c177",
  tabdpt: "#9ece6a",
  tabfm: "#bb9af7",
  mlp: "#7dcfff",
  xgboost: "#f7768e",
};
const DTYPE_COLORS: Record<string, string> = {
  fp32: "#6ea8fe",
  fp16: "#f6c177",
  bf16: "#9ece6a",
};
const GPU_COLORS: Record<string, string> = {
  T4: "#6ea8fe",
  L4: "#f6c177",
  A100: "#9ece6a",
  H100: "#bb9af7",
  B200: "#f7768e",
};
const SPREAD_METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "demographic_parity_diff", label: "Demographic parity diff" },
  { key: "equal_opportunity_diff", label: "Equal opportunity diff" },
  { key: "equalized_odds_diff", label: "Equalized odds diff" },
] as const;
type MetricKey = (typeof SPREAD_METRICS)[number]["key"];

// disparity metric key -> by_attribute criterion block.
const CRITERION: Record<string, "demographic_parity" | "equal_opportunity" | "equalized_odds"> = {
  demographic_parity_diff: "demographic_parity",
  equal_opportunity_diff: "equal_opportunity",
  equalized_odds_diff: "equalized_odds",
};

/** Metric value for a row, honouring the selected sensitive attribute when
 *  the row carries a by_attribute breakdown; falls back to the flat key. */
function metricValue(row: SweepRow, metric: MetricKey, attr: string): number {
  if (metric === "accuracy") return row.accuracy;
  const crit = CRITERION[metric];
  const blk = row.by_attribute?.[attr];
  if (crit && blk) return blk[crit].diff;
  return (row[metric] as number) ?? NaN;
}

/** Standard error of a row's metric.
 *  - accuracy: Wald SE  sqrt(p(1-p)/n), n = test size (sum of subgroup counts);
 *  - disparities: the Wald `diff_se` computed in tabular_exp/fairness.py
 *    (by_attribute[attr][crit].diff_se, or the flat `<metric>_se`). */
function metricSE(row: SweepRow, metric: MetricKey, attr: string): number {
  if (metric === "accuracy") {
    const n = Object.values(row.subgroup_count ?? {}).reduce((a, b) => a + b, 0);
    const p = row.accuracy;
    return n > 0 ? Math.sqrt(Math.max(p * (1 - p), 0) / n) : NaN;
  }
  const crit = CRITERION[metric];
  const blk = row.by_attribute?.[attr];
  const se = crit && blk ? blk[crit].diff_se : undefined;
  return typeof se === "number"
    ? se
    : (row[`${metric}_se` as keyof SweepRow] as number) ?? NaN;
}

/** SE of a (max − min) spread, propagated from the two extreme configs. */
function spreadSE(pairs: { v: number; se: number }[]): number {
  const clean = pairs.filter((p) => Number.isFinite(p.v));
  if (clean.length < 2) return NaN;
  const hi = clean.reduce((a, b) => (b.v > a.v ? b : a));
  const lo = clean.reduce((a, b) => (b.v < a.v ? b : a));
  const sh = Number.isFinite(hi.se) ? hi.se : 0;
  const sl = Number.isFinite(lo.se) ? lo.se : 0;
  return Math.sqrt(sh * sh + sl * sl);
}

function attributesIn(rows: SweepRow[]): string[] {
  const s = new Set<string>();
  for (const r of rows) for (const a of Object.keys(r.by_attribute ?? {})) s.add(a);
  return [...s].sort();
}

const sortBy = <T,>(arr: T[], order: string[], key: (t: T) => string) =>
  [...arr].sort((a, b) => order.indexOf(key(a)) - order.indexOf(key(b)));
const uniq = (xs: string[]) => [...new Set(xs)];

// Models excluded from every chart for now.
const HIDDEN_MODELS = new Set(["tabfm", "xgboost"]);
const spread = (xs: number[]) => (xs.length ? Math.max(...xs) - Math.min(...xs) : 0);

/* ---- e-value helpers ---- */

const evNum = (v: Ev | undefined): number =>
  v === "inf" ? Infinity : typeof v === "number" ? v : NaN;

/** Compact e-value: "∞", scientific for large/small, plain otherwise. */
function fmtE(v: Ev | undefined): string {
  const n = evNum(v);
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "−∞";
  if (n === 0) return "0";
  if (n >= 1e4 || n < 1e-3) return n.toExponential(1);
  return n.toFixed(2);
}

/** log10(E), the number shown in a heat cell (rejection threshold ≈ 1.3). */
function fmtLog10(v: Ev | undefined): string {
  const n = evNum(v);
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "−∞";
  if (n <= 0) return "−∞";
  const l = Math.log10(n);
  return (l >= 0 ? "+" : "") + l.toFixed(1);
}

/** 0..1 colour strength from the e-value magnitude (log-scaled, saturates ~1e30). */
function heatStrength(v: Ev | undefined, reject: boolean): number {
  const n = evNum(v);
  if (!Number.isFinite(n)) return reject ? 1 : 0.15;
  if (reject) return Math.max(0, Math.min(1, Math.log10(Math.max(n, 1)) / 30));
  return Math.max(0, Math.min(0.4, -Math.log10(Math.max(n, 1e-12)) / 20));
}

interface HeatCell {
  text: string;
  reject: boolean;
  strength: number;
  title: string;
}

function Heat({
  rows,
  cols,
  cell,
  rowLabel = "",
}: {
  rows: string[];
  cols: string[];
  cell: (r: string, c: string) => HeatCell | null;
  rowLabel?: string;
}) {
  return (
    <div
      className="heat"
      style={{ gridTemplateColumns: `130px repeat(${cols.length}, minmax(56px, 1fr))` }}
    >
      <div className="heat-corner">{rowLabel}</div>
      {cols.map((c) => (
        <div key={c} className="heat-col">
          {c}
        </div>
      ))}
      {rows.map((r) => (
        <Fragment key={r}>
          <div className="heat-row">{r}</div>
          {cols.map((c) => {
            const d = cell(r, c);
            if (!d)
              return (
                <div key={c} className="heat-cell heat-empty" title="no data">
                  &middot;
                </div>
              );
            const bg = d.reject
              ? `rgba(247,118,142,${(0.16 + 0.62 * d.strength).toFixed(3)})`
              : `rgba(110,168,254,${(0.04 + 0.28 * d.strength).toFixed(3)})`;
            return (
              <div
                key={c}
                className={`heat-cell${d.reject ? " heat-rej" : ""}`}
                style={{ background: bg }}
                title={d.title}
              >
                {d.text}
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}

interface ApiData {
  sweep: Record<string, SweepRow[]>;
  correction: Record<string, CorrectionResult[]>;
  drift: DriftCorrelation[];
  evalues: Evalues;
}

export default function Page() {
  const [data, setData] = useState<ApiData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/results").then((r) => r.json()),
      fetch("/api/correction").then((r) => r.json()),
      fetch("/api/drift").then((r) => r.json()),
      fetch("/api/evalues").then((r) => r.json()),
    ])
      .then(([res, corr, drift, evalues]) => {
        if (res.error) throw new Error(res.error);
        const rawSweep: Record<string, SweepRow[]> = res.datasets ?? {};
        const sweep = Object.fromEntries(
          Object.entries(rawSweep).map(([d, rows]) => [
            d,
            rows.filter((r) => !HIDDEN_MODELS.has(r.model)),
          ]),
        );
        const rawCorr: Record<string, CorrectionResult[]> = corr.datasets ?? {};
        const correction = Object.fromEntries(
          Object.entries(rawCorr)
            .map(([d, list]) => [
              d,
              list.filter((c) => !HIDDEN_MODELS.has(c.model)),
            ] as const)
            .filter(([, list]) => list.length),
        );
        const driftModels: DriftCorrelation[] = (drift.models ?? []).filter(
          (d: DriftCorrelation) => !HIDDEN_MODELS.has(d.model),
        );
        const ev: Evalues = {
          logit: evalues?.logit ?? null,
          metric: evalues?.metric ?? null,
        };
        setData({ sweep, correction, drift: driftModels, evalues: ev });
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  return (
    <main>
      <h1>Tabular hardware-fairness results</h1>
      <p className="sub">
        How much does accuracy move just from changing GPU type or float
        precision, with weights, data and seed held fixed?
      </p>
      {error && <p className="error">Failed to load results: {error}</p>}
      {!data && !error && <p className="sub">Loading&hellip;</p>}
      {data && <Dashboard data={data} />}
    </main>
  );
}

function Dashboard({ data }: { data: ApiData }) {
  const datasets = Object.keys(data.sweep).sort();
  return (
    <>
      <EvalueLogitView level={data.evalues.logit} />
      <EvalueMetricView level={data.evalues.metric} />
      <SpreadOverview sweep={data.sweep} datasets={datasets} />
      <CrossGpuByPrecision sweep={data.sweep} datasets={datasets} />
      <ConfigBreakdown sweep={data.sweep} datasets={datasets} />
      <DriftCorrelationView drift={data.drift} />
      <CorrectionParetoView correction={data.correction} />
      <CorrectionView correction={data.correction} />
    </>
  );
}

/* ---- Correction-vector Pareto: accuracy vs cross-GPU accuracy spread ---- */

interface ParetoPoint {
  x: number; // accuracy spread (max − min across GPUs)
  y: number; // mean accuracy across GPUs
  label: string; // "w=0.5" or "baseline"
  dtype: string;
  coeff: number | number[];
  baseline: boolean;
}

function CorrectionParetoView({
  correction,
}: {
  correction: Record<string, CorrectionResult[]>;
}) {
  const datasets = Object.keys(correction).sort();
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const [model, setModel] = useState("");

  const results = correction[dataset] ?? [];
  const models = uniq(results.map((r) => r.model));
  const activeModel = models.includes(model) ? model : models[0] ?? "";
  const forModel = sortBy(
    results.filter((r) => r.model === activeModel),
    DTYPE_ORDER,
    (r) => r.dtype,
  );

  if (!datasets.length) {
    return (
      <section>
        <h2>Correction-vector Pareto: accuracy vs cross-GPU spread</h2>
        <p className="caption">
          No correction results yet &mdash; run <code>just run-quick</code>.
        </p>
      </section>
    );
  }

  const seriesByDtype: Record<string, ParetoPoint[]> = {};
  const baselinePts: ParetoPoint[] = [];
  for (const res of forModel) {
    seriesByDtype[res.dtype] = [...res.corrected]
      .sort((a, b) => a.acc_weight - b.acc_weight)
      .map((c) => ({
        x: Number(c.accuracy_spread.toFixed(5)),
        y: Number(c.mean_accuracy.toFixed(5)),
        label: `w=${c.acc_weight}`,
        dtype: res.dtype,
        coeff: c.correction_coeff,
        baseline: false,
      }));
    baselinePts.push({
      x: Number(res.baseline.accuracy_spread.toFixed(5)),
      y: Number(res.baseline.mean_accuracy.toFixed(5)),
      label: "baseline",
      dtype: res.dtype,
      coeff: 0,
      baseline: true,
    });
  }

  const allPts = [...Object.values(seriesByDtype).flat(), ...baselinePts];
  // Pareto-optimal = minimise spread, maximise accuracy (up and to the left).
  const optimal = allPts.filter(
    (p) =>
      !allPts.some(
        (q) =>
          q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y),
      ),
  );

  const xs = allPts.map((p) => p.x);
  const ys = allPts.map((p) => p.y);
  const xPad = Math.max(1e-4, (Math.max(...xs) - Math.min(...xs)) * 0.18);
  const yPad = Math.max(1e-4, (Math.max(...ys) - Math.min(...ys)) * 0.18);
  const fmtAxis = (v: number) => (v < 0.01 ? v.toExponential(1) : v.toFixed(3));

  const dtypes = DTYPE_ORDER.filter((dt) => seriesByDtype[dt]?.length);

  return (
    <section>
      <h2>Correction-vector Pareto: accuracy vs cross-GPU spread</h2>
      <div className="controls">
        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select value={activeModel} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
      </div>
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart margin={{ top: 8, right: 24, bottom: 28, left: 12 }}>
          <CartesianGrid stroke="#2a2f3a" />
          <XAxis
            type="number"
            dataKey="x"
            stroke="#9aa0ad"
            fontSize={12}
            domain={[
              Number(Math.max(0, Math.min(...xs) - xPad).toFixed(5)),
              Number((Math.max(...xs) + xPad).toFixed(5)),
            ]}
            tickFormatter={fmtAxis}
            label={{
              value: "accuracy spread across GPUs (max − min)  ← better",
              position: "insideBottom",
              offset: -14,
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            stroke="#9aa0ad"
            fontSize={12}
            domain={[
              Number((Math.min(...ys) - yPad).toFixed(5)),
              Number((Math.max(...ys) + yPad).toFixed(5)),
            ]}
            tickFormatter={(v) => v.toFixed(3)}
            label={{
              value: "mean accuracy  ↑ better",
              angle: -90,
              position: "insideLeft",
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <ZAxis range={[55, 55]} />
          <Tooltip
            cursor={{ stroke: "#2a2f3a" }}
            content={({ payload }) => {
              const p = payload?.[0]?.payload as ParetoPoint | undefined;
              if (!p) return null;
              const coeff =
                typeof p.coeff === "number"
                  ? p.coeff.toFixed(3)
                  : `[${p.coeff.map((v) => v.toFixed(2)).join(", ")}]`;
              return (
                <div
                  style={{
                    background: "#181b22",
                    border: "1px solid #2a2f3a",
                    padding: "6px 8px",
                    fontSize: 12,
                  }}
                >
                  <div>
                    {p.dtype} &middot; {p.label}
                  </div>
                  <div>mean acc: {p.y.toFixed(4)}</div>
                  <div>acc spread: {p.x.toFixed(4)}</div>
                  {!p.baseline && <div>correction C: {coeff}</div>}
                </div>
              );
            }}
          />
          <Legend />
          {dtypes.map((dt) => (
            <Scatter
              key={dt}
              name={dt}
              data={seriesByDtype[dt]}
              fill={DTYPE_COLORS[dt] ?? "#8a8f9c"}
              line={{ stroke: DTYPE_COLORS[dt] ?? "#8a8f9c", strokeWidth: 1.5 }}
              lineType="joint"
            />
          ))}
          <Scatter
            name="baseline (c = 0)"
            data={baselinePts}
            fill="#565a66"
            shape="cross"
          />
          <Scatter
            name="Pareto-optimal"
            data={optimal}
            fill="#e6e8ec"
            shape="diamond"
          />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="caption">
        Each curve is one float precision; points are the correction fit at{" "}
        <code>acc_weight</code> 0 &rarr; 1 (see the trade-off knob below), joined
        in that order. Grey crosses are the uncorrected <code>c = 0</code>{" "}
        baseline; white diamonds are Pareto-optimal across everything shown
        (nothing has both higher accuracy and lower spread). Up-and-to-the-left
        is better: high mean accuracy, low cross-GPU accuracy spread.
      </p>
    </section>
  );
}

/* ---- Softmax-input (logit-level) drift, e-value test ---- */

function EvalueLogitView({ level }: { level: EvalueLevel | null }) {
  const [ds, setDs] = useState("__all__");

  const datasets = useMemo(
    () => (level ? uniq(level.per_dataset.map((r) => r.dataset)).sort() : []),
    [level],
  );
  const models = useMemo(
    () => (level ? uniq(level.combined.map((r) => r.model)).sort() : []),
    [level],
  );
  const gpus = useMemo(
    () =>
      level
        ? sortBy(uniq(level.combined.map((r) => r.gpu)), GPU_ORDER, (x) => x)
        : [],
    [level],
  );

  if (!level || !level.combined.length) {
    return (
      <section>
        <h2>Softmax-input drift between GPUs (e-value test)</h2>
        <p className="caption">
          No <code>results/evalues/logit_level.json</code> yet &mdash; run{" "}
          <code>just run-quick</code> (or <code>just evalues --logit-only</code>{" "}
          after a logit capture).
        </p>
      </section>
    );
  }

  const perDs =
    ds === "__all__"
      ? null
      : level.per_dataset.filter((r) => r.dataset === ds);
  const nRej = level.combined.filter((r) => r["ebh_reject@0.05"]).length;

  const cell = (m: string, g: string): HeatCell | null => {
    if (perDs) {
      const r = perDs.find((x) => x.model === m && x.gpu === g);
      if (!r) return null;
      const reject = evNum(r.evalue) >= 1 / level.alpha;
      return {
        text: fmtLog10(r.evalue),
        reject,
        strength: heatStrength(r.evalue, reject),
        title:
          `${m} / ${g} @ ${ds}\n` +
          `E = ${fmtE(r.evalue)}  (${reject ? "≥ 1/α → significant" : "not significant"})\n` +
          `mean |Δ log-odds| = ${r.mean_abs_logodds_drift?.toExponential(2)}\n` +
          `max |Δ log-odds| = ${r.max_abs_logodds_drift?.toExponential(2)}\n` +
          `prediction flips vs T4 = ${r.argmax_flips_vs_ref}\n` +
          `noise floor: ${r.noise_kind ?? (r.deterministic ? "arithmetic" : "within-GPU repeats")}`,
      };
    }
    const r = level.combined.find((x) => x.model === m && x.gpu === g);
    if (!r) return null;
    return {
      text: fmtLog10(r.evalue_product),
      reject: r["ebh_reject@0.05"],
      strength: heatStrength(r.evalue_product, r["ebh_reject@0.05"]),
      title:
        `${m} / ${g} — product over ${datasets.length} datasets\n` +
        `E = ${fmtE(r.evalue_product)}\n` +
        `e-BH @ α=0.05: ${r["ebh_reject@0.05"] ? "reject (drift detected)" : "—"}\n` +
        `e-BH @ α=0.10: ${r["ebh_reject@0.1"] ? "reject" : "—"}`,
    };
  };

  return (
    <section>
      <h2>Softmax-input drift between GPUs (e-value test)</h2>
      <div className="controls">
        <label>
          Datasets
          <select value={ds} onChange={(e) => setDs(e.target.value)}>
            <option value="__all__">all (evidence multiplied)</option>
            {datasets.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <span className="pill pill-hot">
          {nRej}/{level.combined.length} model&times;GPU reject e-BH @ 0.05
        </span>
      </div>
      <Heat rows={models} cols={gpus} cell={cell} rowLabel="log₁₀ E" />
      <p className="caption">
        H&#8320;: GPU <em>g</em> and T4 produce the same per-row log-odds (softmax
        input), up to noise. Cell = log&#8321;&#8320; of the e-value; red =
        rejected{" "}
        {perDs ? (
          <>
            (Ville bound, E &ge; 1/&alpha; = {(1 / level.alpha).toFixed(0)}, this
            dataset only)
          </>
        ) : (
          <>
            by e-BH across the {level.combined.length}-cell grid (FDR &le;{" "}
            {level.alpha})
          </>
        )}
        . Hover for drift magnitude and prediction flips. Repeated passes are
        bit-identical for these models at fp32, so the noise floor is
        arithmetic-precision, not stochastic.
      </p>
    </section>
  );
}

/* ---- Fairness-metric drift between GPUs, e-value test ---- */

const METRIC_CRITERIA = [
  { key: "demographic_parity", label: "Demographic parity gap" },
  { key: "equal_opportunity", label: "Equal opportunity gap" },
  { key: "equalized_odds", label: "Equalized odds gap" },
  { key: "mcnemar", label: "McNemar per subgroup" },
] as const;

function EvalueMetricView({ level }: { level: EvalueLevel | null }) {
  const [dtype, setDtype] = useState("fp32");
  const [attr, setAttr] = useState("gender");
  const [crit, setCrit] = useState<string>("demographic_parity");

  const dtypes = useMemo(
    () =>
      level
        ? sortBy(
            uniq(level.combined.map((r) => r.dtype ?? "")),
            DTYPE_ORDER,
            (x) => x,
          )
        : [],
    [level],
  );
  const attrs = useMemo(
    () =>
      level ? uniq(level.combined.map((r) => r.attribute ?? "")).sort() : [],
    [level],
  );
  const models = useMemo(
    () => (level ? uniq(level.combined.map((r) => r.model)).sort() : []),
    [level],
  );
  const gpus = useMemo(
    () =>
      level
        ? sortBy(uniq(level.combined.map((r) => r.gpu)), GPU_ORDER, (x) => x)
        : [],
    [level],
  );

  if (!level || !level.combined.length) {
    return (
      <section>
        <h2>Fairness-metric drift between GPUs (e-value test)</h2>
        <p className="caption">
          No <code>results/evalues/metric_level.json</code> yet &mdash; run{" "}
          <code>just run-quick</code>.
        </p>
      </section>
    );
  }

  const activeDtype = dtypes.includes(dtype) ? dtype : dtypes[0] ?? "";
  const activeAttr = attrs.includes(attr) ? attr : attrs[0] ?? "";
  const rows = level.combined.filter(
    (r) => r.dtype === activeDtype && r.attribute === activeAttr,
  );
  const nRej = level.combined.filter((r) => r["ebh_reject@0.05"]).length;

  const cell = (m: string, g: string): HeatCell | null => {
    if (crit === "mcnemar") {
      const subs = rows.filter(
        (r) => r.model === m && r.gpu === g && r.test.startsWith("mcnemar["),
      );
      if (!subs.length) return null;
      const rej = subs.filter((r) => r["ebh_reject@0.05"]);
      const worst = subs.reduce((a, b) =>
        evNum(b.evalue_product) > evNum(a.evalue_product) ? b : a,
      );
      return {
        text: `${rej.length}/${subs.length}`,
        reject: rej.length > 0,
        strength: heatStrength(worst.evalue_product, rej.length > 0),
        title:
          `${m} / ${g} — ${activeAttr}, McNemar flip-symmetry per subgroup\n` +
          subs
            .map(
              (s) =>
                `  ${s.test.slice(8, -1)}: E = ${fmtE(s.evalue_product)}${
                  s["ebh_reject@0.05"] ? "  ✗ reject" : ""
                }`,
            )
            .join("\n"),
      };
    }
    const r = rows.find(
      (x) => x.model === m && x.gpu === g && x.test === crit,
    );
    if (!r) return null;
    return {
      text: fmtLog10(r.evalue_product),
      reject: r["ebh_reject@0.05"],
      strength: heatStrength(r.evalue_product, r["ebh_reject@0.05"]),
      title:
        `${m} / ${g} — ${activeAttr}, ${crit}\n` +
        `E = ${fmtE(r.evalue_product)}  (product over datasets)\n` +
        `e-BH @ 0.05: ${r["ebh_reject@0.05"] ? "reject (drift detected)" : "—"}`,
    };
  };

  return (
    <section>
      <h2>Fairness-metric drift between GPUs (e-value test)</h2>
      <div className="controls">
        <label>
          Criterion
          <select value={crit} onChange={(e) => setCrit(e.target.value)}>
            {METRIC_CRITERIA.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Float type
          <select
            value={activeDtype}
            onChange={(e) => setDtype(e.target.value)}
          >
            {dtypes.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Sensitive attribute
          <select value={activeAttr} onChange={(e) => setAttr(e.target.value)}>
            {attrs.map((a) => (
              <option key={a}>{a}</option>
            ))}
          </select>
        </label>
        <span className="pill pill-hot">
          {nRej}/{level.combined.length} hypotheses reject e-BH @ 0.05
        </span>
      </div>
      <Heat
        rows={models}
        cols={gpus}
        cell={cell}
        rowLabel={crit === "mcnemar" ? "reject / n" : "log₁₀ E"}
      />
      <p className="caption">
        H&#8320;: GPU <em>g</em> and T4 induce the same fairness metric on the
        test distribution (paired per-row test; same rows, same weights). Each
        cell&rsquo;s e-value is the product over the datasets that expose{" "}
        <em>{activeAttr}</em>; red = rejected by e-BH across the full{" "}
        {level.combined.length}-hypothesis grid (FDR &le; {level.alpha}).
        {crit === "mcnemar"
          ? " Cell shows how many of the attribute's subgroups have asymmetric prediction flips; hover for each."
          : " Gap-drift e-value on the signed max−min subgroup gap."}
      </p>
    </section>
  );
}

/* ---- Drift vs fairness-shift correlation (T4 reference) ---- */

function DriftCorrelationView({ drift }: { drift: DriftCorrelation[] }) {
  const [model, setModel] = useState("");
  if (!drift.length) {
    return (
      <section>
        <h2>Logit drift vs fairness shift (T4 reference)</h2>
        <p className="caption">
          No drift-correlation results yet &mdash; run{" "}
          <code>just drift-correlation</code>.
        </p>
      </section>
    );
  }
  const active = drift.find((d) => d.model === model) ?? drift[0];
  const chartData = active.points.map((p) => ({
    ...p,
    label: `${p.dataset} / ${p.gpu}`,
  }));

  return (
    <section>
      <h2>Logit drift vs fairness shift (T4 reference)</h2>
      <div className="controls">
        <label>
          Model
          <select
            value={active.model}
            onChange={(e) => setModel(e.target.value)}
          >
            {drift.map((d) => (
              <option key={d.model}>{d.model}</option>
            ))}
          </select>
        </label>
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" />
          <XAxis
            type="number"
            dataKey="x"
            stroke="#9aa0ad"
            fontSize={12}
            tickFormatter={(v) => v.toFixed(2)}
            label={{
              value: "mean centered-logit L2 drift vs T4",
              position: "insideBottom",
              offset: -12,
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            stroke="#9aa0ad"
            fontSize={12}
            tickFormatter={(v) => v.toFixed(3)}
            label={{
              value: "|dp-diff − T4 dp-diff|",
              angle: -90,
              position: "insideLeft",
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <ZAxis range={[60, 60]} />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #2a2f3a" }}
            formatter={(v: number) => v.toFixed(4)}
            labelFormatter={() => ""}
            content={({ payload }) => {
              const p = payload?.[0]?.payload as
                | (typeof chartData)[number]
                | undefined;
              if (!p) return null;
              return (
                <div
                  style={{
                    background: "#181b22",
                    border: "1px solid #2a2f3a",
                    padding: "6px 8px",
                    fontSize: 12,
                  }}
                >
                  <div>{p.label}</div>
                  <div>drift: {p.x.toFixed(4)}</div>
                  <div>fairness shift: {p.y.toFixed(4)}</div>
                </div>
              );
            }}
          />
          <Scatter data={chartData} fill={MODEL_COLORS[active.model] ?? "#8a8f9c"}>
            {chartData.map((p, i) => (
              <Cell
                key={i}
                fill={
                  p.gpu === "T4"
                    ? "#565a66"
                    : MODEL_COLORS[active.model] ?? "#8a8f9c"
                }
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <p className="caption">
        Each point is one (dataset, GPU) pair for <code>{active.model}</code> at
        fp32. X is how far that GPU&rsquo;s softmax-input logits move from the T4
        run; Y is how far the demographic-parity difference moves.{" "}
        {typeof active.pearson_r === "number" ? (
          <>
            Pearson R = <strong>{active.pearson_r.toFixed(3)}</strong>
            {typeof active.p_value === "number"
              ? ` (p = ${active.p_value.toExponential(2)})`
              : ""}{" "}
            over {active.n_points} points.
          </>
        ) : (
          <>
            Pearson R is undefined (the drift values barely vary across the{" "}
            {active.n_points} points).
          </>
        )}{" "}
        T4-vs-T4 points sit at the origin (grey).
      </p>
    </section>
  );
}

/* ---- Section 1: spread across datasets x models ---- */

function SpreadOverview({
  sweep,
  datasets,
}: {
  sweep: Record<string, SweepRow[]>;
  datasets: string[];
}) {
  const [metric, setMetric] = useState<MetricKey>("accuracy");
  const [attr, setAttr] = useState("");
  const [dtype, setDtype] = useState("all");
  const models = useMemo(
    () => uniq(datasets.flatMap((d) => sweep[d].map((r) => r.model))).sort(),
    [sweep, datasets],
  );
  const allRows = useMemo(
    () => datasets.flatMap((d) => sweep[d]),
    [sweep, datasets],
  );
  const dtypes = useMemo(
    () => sortBy(uniq(allRows.map((r) => r.dtype)), DTYPE_ORDER, (x) => x),
    [allRows],
  );
  const activeDtype = dtype === "all" || dtypes.includes(dtype) ? dtype : "all";
  const attrs = useMemo(() => attributesIn(allRows), [allRows]);
  const activeAttr = attrs.includes(attr) ? attr : attrs[0] ?? "";
  const showAttr = metric !== "accuracy" && attrs.length > 0;

  const chartData = datasets.map((dataset) => {
    const row: Record<string, string | number> = { dataset };
    for (const model of models) {
      const pairs = sweep[dataset]
        .filter(
          (r) =>
            r.model === model &&
            (activeDtype === "all" || r.dtype === activeDtype),
        )
        .map((r) => ({
          v: metricValue(r, metric, activeAttr),
          se: metricSE(r, metric, activeAttr),
        }))
        .filter((p) => typeof p.v === "number" && !Number.isNaN(p.v));
      if (pairs.length) {
        row[model] = Number(spread(pairs.map((p) => p.v)).toFixed(4));
        const se = spreadSE(pairs);
        if (Number.isFinite(se)) row[`${model}__se`] = Number(se.toFixed(4));
      }
    }
    return row;
  });

  return (
    <section>
      <h2>Metric spread across GPU &times; float type</h2>
      <div className="controls">
        <label>
          Metric
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
          >
            {SPREAD_METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Float type
          <select value={activeDtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="all">all (GPU &times; dtype)</option>
            {dtypes.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        {showAttr && (
          <label>
            Sensitive attribute
            <select
              value={activeAttr}
              onChange={(e) => setAttr(e.target.value)}
            >
              {attrs.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" vertical={false} />
          <XAxis dataKey="dataset" stroke="#9aa0ad" fontSize={12} />
          <YAxis
            stroke="#9aa0ad"
            fontSize={12}
            tickFormatter={(v) => v.toFixed(3)}
            label={{
              value: "max − min",
              angle: -90,
              position: "insideLeft",
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #2a2f3a" }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend />
          {models.map((model) => (
            <Bar
              key={model}
              dataKey={model}
              fill={MODEL_COLORS[model] ?? "#8a8f9c"}
            >
              <ErrorBar
                dataKey={`${model}__se`}
                width={3}
                strokeWidth={1.5}
                stroke="#e6e8ec"
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        Bar height is the range (max − min) of the chosen metric over{" "}
        {activeDtype === "all" ? (
          <>every GPU&times;dtype run</>
        ) : (
          <>
            every GPU run at <code>{activeDtype}</code>
          </>
        )}{" "}
        for that model and dataset. A near-zero bar (e.g.
        XGBoost) is the deterministic control. Error bars propagate the standard
        error of the two extreme configs (&radic;(SE&sup2;<sub>max</sub> +
        SE&sup2;<sub>min</sub>)) &mdash; Wald SE for accuracy, the analytic{" "}
        <code>diff_se</code> for disparities.
      </p>
    </section>
  );
}

/* ---- Section 1b: cross-GPU spread, holding float precision fixed ---- */

function CrossGpuByPrecision({
  sweep,
  datasets,
}: {
  sweep: Record<string, SweepRow[]>;
  datasets: string[];
}) {
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const [metric, setMetric] = useState<MetricKey>("accuracy");
  const [attr, setAttr] = useState("");

  const rows = sweep[dataset] ?? [];
  const models = useMemo(() => uniq(rows.map((r) => r.model)).sort(), [rows]);
  const dtypes = sortBy(uniq(rows.map((r) => r.dtype)), DTYPE_ORDER, (x) => x);
  const attrs = useMemo(() => attributesIn(rows), [rows]);
  const activeAttr = attrs.includes(attr) ? attr : attrs[0] ?? "";
  const showAttr = metric !== "accuracy" && attrs.length > 0;

  const chartData = dtypes.map((dt) => {
    const row: Record<string, string | number> = { dtype: dt };
    for (const model of models) {
      const pairs = rows
        .filter((r) => r.model === model && r.dtype === dt)
        .map((r) => ({
          v: metricValue(r, metric, activeAttr),
          se: metricSE(r, metric, activeAttr),
        }))
        .filter((p) => typeof p.v === "number" && !Number.isNaN(p.v));
      // Need at least two GPU runs to have a between-GPU spread.
      if (pairs.length > 1) {
        row[model] = Number(spread(pairs.map((p) => p.v)).toFixed(4));
        const se = spreadSE(pairs);
        if (Number.isFinite(se)) row[`${model}__se`] = Number(se.toFixed(4));
      }
    }
    return row;
  });

  return (
    <section>
      <h2>Between-GPU spread within each float precision</h2>
      <div className="controls">
        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Metric
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
          >
            {SPREAD_METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        {showAttr && (
          <label>
            Sensitive attribute
            <select value={activeAttr} onChange={(e) => setAttr(e.target.value)}>
              {attrs.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" vertical={false} />
          <XAxis dataKey="dtype" stroke="#9aa0ad" fontSize={12} />
          <YAxis
            stroke="#9aa0ad"
            fontSize={12}
            domain={[0, 0.5]}
            allowDataOverflow
            tickFormatter={(v) => v.toFixed(3)}
            label={{
              value: "max − min across GPUs",
              angle: -90,
              position: "insideLeft",
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #2a2f3a" }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend />
          {models.map((model) => (
            <Bar
              key={model}
              dataKey={model}
              fill={MODEL_COLORS[model] ?? "#8a8f9c"}
            >
              <ErrorBar
                dataKey={`${model}__se`}
                width={3}
                strokeWidth={1.5}
                stroke="#e6e8ec"
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        For each float precision, the range (max − min) of the metric across GPU
        types, with the model, data and dtype all held fixed &mdash; isolates the
        pure hardware effect from the precision effect. Models with a single GPU
        run in a dtype (e.g. XGBoost) are omitted. Error bars propagate the
        standard error of the two extreme GPU configs.
      </p>
    </section>
  );
}

/* ---- Section 2: per-config metric (GPU x dtype), with SE error bars ---- */

function ConfigBreakdown({
  sweep,
  datasets,
}: {
  sweep: Record<string, SweepRow[]>;
  datasets: string[];
}) {
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const [metric, setMetric] = useState<MetricKey>("accuracy");
  const [attr, setAttr] = useState("");
  const models = useMemo(
    () => uniq((sweep[dataset] ?? []).map((r) => r.model)).sort(),
    [sweep, dataset],
  );
  const [model, setModel] = useState(models[0] ?? "");
  const activeModel = models.includes(model) ? model : models[0] ?? "";

  const rows = (sweep[dataset] ?? []).filter((r) => r.model === activeModel);
  const dtypes = sortBy(uniq(rows.map((r) => r.dtype)), DTYPE_ORDER, (x) => x);
  const gpus = sortBy(uniq(rows.map((r) => r.gpu)), GPU_ORDER, (x) => x);
  const attrs = useMemo(() => attributesIn(rows), [rows]);
  const activeAttr = attrs.includes(attr) ? attr : attrs[0] ?? "";
  const showAttr = metric !== "accuracy" && attrs.length > 0;

  // Grouped by float type, one bar per GPU inside each group.
  const chartData = dtypes.map((dt) => {
    const row: Record<string, string | number> = { dtype: dt };
    for (const gpu of gpus) {
      const hit = rows.find((r) => r.gpu === gpu && r.dtype === dt);
      if (!hit) continue;
      const v = metricValue(hit, metric, activeAttr);
      if (typeof v !== "number" || Number.isNaN(v)) continue;
      row[gpu] = Number(v.toFixed(5));
      const se = metricSE(hit, metric, activeAttr);
      if (Number.isFinite(se)) row[`${gpu}__se`] = Number(se.toFixed(5));
    }
    return row;
  });

  const vals = rows
    .map((r) => metricValue(r, metric, activeAttr))
    .filter((v) => typeof v === "number" && !Number.isNaN(v));
  const pad = Math.max(metric === "accuracy" ? 0.01 : 0.005, spread(vals) * 0.3);
  const domain: [number, number] = vals.length
    ? [0, Number((Math.max(...vals) + pad).toFixed(4))]
    : [0, 1];

  return (
    <section>
      <h2>Metric per float type, grouped by GPU</h2>
      <div className="controls">
        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select
            value={activeModel}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Metric
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
          >
            {SPREAD_METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        {showAttr && (
          <label>
            Sensitive attribute
            <select value={activeAttr} onChange={(e) => setAttr(e.target.value)}>
              {attrs.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" vertical={false} />
          <XAxis dataKey="dtype" stroke="#9aa0ad" fontSize={12} />
          <YAxis
            stroke="#9aa0ad"
            fontSize={12}
            domain={domain}
            tickFormatter={(v) => v.toFixed(3)}
          />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #2a2f3a" }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend />
          {gpus.map((gpu) => (
            <Bar key={gpu} dataKey={gpu} fill={GPU_COLORS[gpu] ?? "#8a8f9c"}>
              <ErrorBar
                dataKey={`${gpu}__se`}
                width={3}
                strokeWidth={1.5}
                stroke="#e6e8ec"
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        Bars are grouped by float type, one per GPU inside each group, so the
        cross-GPU comparison at a fixed precision is the visual unit. Y-axis
        starts at 0. Error bars are &plusmn;1 standard error &mdash; Wald{" "}
        <code>&radic;(p(1&minus;p)/n)</code> for accuracy, the analytic{" "}
        <code>diff_se</code> from <code>tabular_exp/fairness.py</code> for the
        disparity gaps. Bars whose error bars all overlap within a group mean
        the hardware-induced difference is within sampling noise.
      </p>
    </section>
  );
}

/* ---- Section 3: correction-vector intervention ---- */

function CorrectionView({
  correction,
}: {
  correction: Record<string, CorrectionResult[]>;
}) {
  const datasets = Object.keys(correction).sort();
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const [model, setModel] = useState("");
  const [dtype, setDtype] = useState("");

  const results = correction[dataset] ?? [];
  const models = uniq(results.map((r) => r.model));
  const activeModel = models.includes(model) ? model : models[0] ?? "";
  const dtypesForModel = sortBy(
    uniq(results.filter((r) => r.model === activeModel).map((r) => r.dtype)),
    DTYPE_ORDER,
    (x) => x,
  );
  const activeDtype = dtypesForModel.includes(dtype)
    ? dtype
    : dtypesForModel[0] ?? "";
  const active = results.find(
    (r) => r.model === activeModel && r.dtype === activeDtype,
  );

  if (!datasets.length) {
    return (
      <section>
        <h2>Correction-vector intervention</h2>
        <p className="caption">
          No correction results yet &mdash; run{" "}
          <code>just run-quick</code> or{" "}
          <code>modal run modal_app.py::correction</code>.
        </p>
      </section>
    );
  }

  const chartData = active
    ? active.corrected.map((c) => ({
        acc_weight: c.acc_weight,
        mean_accuracy: c.mean_accuracy,
        accuracy_spread: c.accuracy_spread,
      }))
    : [];

  return (
    <section>
      <h2>Correction-vector intervention</h2>
      <div className="controls">
        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select value={activeModel} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Float type
          <select value={activeDtype} onChange={(e) => setDtype(e.target.value)}>
            {dtypesForModel.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" />
          <XAxis
            dataKey="acc_weight"
            type="number"
            stroke="#9aa0ad"
            fontSize={12}
            label={{
              value: "acc_weight",
              position: "insideBottom",
              offset: -4,
              fill: "#9aa0ad",
              fontSize: 12,
            }}
          />
          <YAxis stroke="#9aa0ad" fontSize={12} tickFormatter={(v) => v.toFixed(3)} />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #2a2f3a" }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend />
          {active && (
            <>
              <ReferenceLine
                y={active.baseline.mean_accuracy}
                stroke="#6ea8fe"
                strokeDasharray="4 4"
                label={{
                  value: "baseline mean acc",
                  fill: "#6ea8fe",
                  fontSize: 11,
                  position: "insideTopRight",
                }}
              />
              <ReferenceLine
                y={active.baseline.accuracy_spread}
                stroke="#f7768e"
                strokeDasharray="4 4"
                label={{
                  value: "baseline spread",
                  fill: "#f7768e",
                  fontSize: 11,
                  position: "insideBottomRight",
                }}
              />
            </>
          )}
          <Line
            type="monotone"
            dataKey="mean_accuracy"
            stroke="#6ea8fe"
            strokeWidth={2}
            dot
          />
          <Line
            type="monotone"
            dataKey="accuracy_spread"
            stroke="#f7768e"
            strokeWidth={2}
            dot
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="caption">
        A single softmax-input correction vector, shared across every GPU, is fit
        on the dedicated 20% calibration split. <code>acc_weight</code> traces
        the trade-off: higher weight keeps accuracy up, lower weight pushes the
        cross-GPU accuracy spread toward zero. Dashed lines are the uncorrected
        (c = 0) baseline
        {active ? ` (n_test=${active.n_test}, n_calib=${active.n_calib}).` : "."}
        {active?.mean_l2_logit_drift_vs_t4
          ? ` Mean centered-logit L2 drift vs ${active.reference_gpu ?? "T4"}: ` +
            active.gpus
              .map(
                (g) =>
                  `${g} ${(active.mean_l2_logit_drift_vs_t4?.[g] ?? 0).toFixed(3)}`,
              )
              .join(", ") +
            "."
          : ""}
      </p>
    </section>
  );
}

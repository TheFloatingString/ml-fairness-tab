"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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

interface ApiData {
  sweep: Record<string, SweepRow[]>;
  correction: Record<string, CorrectionResult[]>;
  drift: DriftCorrelation[];
}

export default function Page() {
  const [data, setData] = useState<ApiData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/results").then((r) => r.json()),
      fetch("/api/correction").then((r) => r.json()),
      fetch("/api/drift").then((r) => r.json()),
    ])
      .then(([res, corr, drift]) => {
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
        setData({ sweep, correction, drift: driftModels });
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
      <SpreadOverview sweep={data.sweep} datasets={datasets} />
      <CrossGpuByPrecision sweep={data.sweep} datasets={datasets} />
      <ConfigBreakdown sweep={data.sweep} datasets={datasets} />
      <DriftCorrelationView drift={data.drift} />
      <CorrectionView correction={data.correction} />
    </>
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
        run; Y is how far the demographic-parity difference moves. Pearson R ={" "}
        <strong>{active.pearson_r.toFixed(3)}</strong> (p ={" "}
        {active.p_value.toExponential(2)}) over {active.n_points} points. T4-vs-T4
        points sit at the origin (grey).
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
  const models = useMemo(
    () => uniq(datasets.flatMap((d) => sweep[d].map((r) => r.model))).sort(),
    [sweep, datasets],
  );
  const allRows = useMemo(
    () => datasets.flatMap((d) => sweep[d]),
    [sweep, datasets],
  );
  const attrs = useMemo(() => attributesIn(allRows), [allRows]);
  const activeAttr = attrs.includes(attr) ? attr : attrs[0] ?? "";
  const showAttr = metric !== "accuracy" && attrs.length > 0;

  const chartData = datasets.map((dataset) => {
    const row: Record<string, string | number> = { dataset };
    for (const model of models) {
      const vals = sweep[dataset]
        .filter((r) => r.model === model)
        .map((r) => metricValue(r, metric, activeAttr))
        .filter((v) => typeof v === "number" && !Number.isNaN(v));
      if (vals.length) row[model] = Number(spread(vals).toFixed(4));
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
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        Bar height is the range (max − min) of the chosen metric over every
        GPU&times;dtype run for that model and dataset. A near-zero bar (e.g.
        XGBoost) is the deterministic control.
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
      const vals = rows
        .filter((r) => r.model === model && r.dtype === dt)
        .map((r) => metricValue(r, metric, activeAttr))
        .filter((v) => typeof v === "number" && !Number.isNaN(v));
      // Need at least two GPU runs to have a between-GPU spread.
      if (vals.length > 1) row[model] = Number(spread(vals).toFixed(4));
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
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        For each float precision, the range (max − min) of the metric across GPU
        types, with the model, data and dtype all held fixed &mdash; isolates the
        pure hardware effect from the precision effect. Models with a single GPU
        run in a dtype (e.g. XGBoost) are omitted.
      </p>
    </section>
  );
}

/* ---- Section 2: per-config accuracy (GPU x dtype) ---- */

function ConfigBreakdown({
  sweep,
  datasets,
}: {
  sweep: Record<string, SweepRow[]>;
  datasets: string[];
}) {
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const models = useMemo(
    () => uniq((sweep[dataset] ?? []).map((r) => r.model)).sort(),
    [sweep, dataset],
  );
  const [model, setModel] = useState(models[0] ?? "");
  const activeModel = models.includes(model) ? model : models[0] ?? "";

  const rows = (sweep[dataset] ?? []).filter((r) => r.model === activeModel);
  const dtypes = sortBy(uniq(rows.map((r) => r.dtype)), DTYPE_ORDER, (x) => x);
  const gpus = sortBy(uniq(rows.map((r) => r.gpu)), GPU_ORDER, (x) => x);

  const chartData = gpus.map((gpu) => {
    const row: Record<string, string | number> = { gpu };
    for (const dt of dtypes) {
      const hit = rows.find((r) => r.gpu === gpu && r.dtype === dt);
      if (hit) row[dt] = hit.accuracy;
    }
    return row;
  });

  const accs = rows.map((r) => r.accuracy);
  const pad = Math.max(0.002, spread(accs) * 0.25);
  const domain: [number, number] = accs.length
    ? [
        Number((Math.min(...accs) - pad).toFixed(4)),
        Number((Math.max(...accs) + pad).toFixed(4)),
      ]
    : [0, 1];

  return (
    <section>
      <h2>Accuracy per GPU and float type</h2>
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
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#2a2f3a" vertical={false} />
          <XAxis dataKey="gpu" stroke="#9aa0ad" fontSize={12} />
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
          {dtypes.map((dt) => (
            <Bar key={dt} dataKey={dt} fill={DTYPE_COLORS[dt] ?? "#8a8f9c"} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="caption">
        Y-axis is zoomed to this model&rsquo;s accuracy range so hardware-induced
        differences are visible.
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

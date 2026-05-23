"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { formatRelative, formatUsdCents } from "@/lib/format";
import styles from "./generate.module.css";

type Status = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELED";

interface RecentItem {
  id: string;
  prompt: string;
  status: string;
  durationSeconds: number;
  resolution: string;
  createdAt: string;
  resultUrl: string | null;
  thumbnailUrl: string | null;
}

interface Props {
  ratePerSecondCents: number;
  initialRecent: RecentItem[];
}

const PRESET_PROMPTS = [
  "Slow push-in on a perfume bottle on warm marble, glass refracts soft window light, label reads AURELIA Nº7 in clean serif, 4 sec, 24 fps.",
  "Tracking shot following a runner's silhouette across a foggy bridge at dawn, breath visible, cinematic anamorphic 2.39:1.",
  "Close-up of a single drop of espresso falling into a porcelain cup; surface crema forms with realistic micro-bubbles, 5 sec.",
];

export function GenerateClient({
  ratePerSecondCents,
  initialRecent,
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [aspect, setAspect] = useState<"16:9" | "9:16" | "1:1" | "4:5">("16:9");
  const [duration, setDuration] = useState<number>(4);
  const [resolution, setResolution] = useState<"720p" | "1080p" | "4k">("1080p");
  const [seed, setSeed] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentItem[]>(initialRecent);

  // Poll for status updates of any non-terminal jobs every 2s.
  useEffect(() => {
    const active = recent.filter(
      (r) => r.status === "QUEUED" || r.status === "RUNNING",
    );
    if (active.length === 0) return;
    const handle = setInterval(async () => {
      const next = await Promise.all(
        recent.map(async (r) => {
          if (r.status !== "QUEUED" && r.status !== "RUNNING") return r;
          const resp = await fetch(`/api/generations/${r.id}`);
          if (!resp.ok) return r;
          const data = (await resp.json()) as { generation?: RecentItem };
          if (!data.generation) return r;
          return {
            ...r,
            status: data.generation.status,
            resultUrl: data.generation.resultUrl,
            thumbnailUrl: data.generation.thumbnailUrl,
          };
        }),
      );
      setRecent(next);
    }, 2000);
    return () => clearInterval(handle);
  }, [recent]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) {
      setError("Prompt is required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    const resp = await fetch("/api/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        prompt: prompt.trim(),
        negativePrompt: negative.trim() || undefined,
        aspectRatio: aspect,
        durationSeconds: duration,
        resolution,
        seed: seed ? Number(seed) : undefined,
      }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({ error: "unknown" }));
      setError(
        body.message ??
          (typeof body.error === "string"
            ? body.error.replaceAll("_", " ")
            : "Unable to submit"),
      );
      setSubmitting(false);
      return;
    }
    const created = await resp.json();
    setRecent((prev) => [
      {
        id: created.id,
        prompt: prompt.trim(),
        status: created.status as Status,
        durationSeconds: duration,
        resolution,
        createdAt: new Date().toISOString(),
        resultUrl: null,
        thumbnailUrl: null,
      },
      ...prev,
    ]);
    setPrompt("");
    setNegative("");
    setSubmitting(false);
  }

  const projectedCost = duration * ratePerSecondCents;

  return (
    <>
      <header className={styles.header}>
        <div>
          <p className="eyebrow">Generate</p>
          <h1 className={styles.title}>New generation</h1>
          <p className={styles.subtitle}>
            Describe the shot. The retrofit ladder handles glyph integrity,
            anatomy, motion, and physics conditioning automatically.
          </p>
        </div>
        <p className={styles.rate}>
          <span className={styles.rateLabel}>Current rate</span>
          <span className={`mono ${styles.rateValue}`}>
            {formatUsdCents(ratePerSecondCents)} / s
          </span>
        </p>
      </header>

      <div className={styles.split}>
        <form className={styles.form} onSubmit={handleSubmit}>
          <label className={styles.field}>
            <span>Prompt</span>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={5}
              placeholder="A slow push-in on a perfume bottle …"
              className={styles.textarea}
            />
          </label>

          <div className={styles.presets}>
            <span className={styles.presetLabel}>Try:</span>
            {PRESET_PROMPTS.map((p) => (
              <button
                key={p}
                type="button"
                className={styles.presetChip}
                onClick={() => setPrompt(p)}
                title={p}
              >
                {p.split(",")[0]}…
              </button>
            ))}
          </div>

          <label className={styles.field}>
            <span>Negative prompt (optional)</span>
            <input
              type="text"
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              placeholder="warped letters, extra fingers, plastic skin"
              className={styles.input}
            />
          </label>

          <div className={styles.row}>
            <label className={styles.field}>
              <span>Aspect ratio</span>
              <select
                value={aspect}
                onChange={(e) => setAspect(e.target.value as typeof aspect)}
                className={styles.select}
              >
                <option value="16:9">16:9 widescreen</option>
                <option value="9:16">9:16 vertical</option>
                <option value="1:1">1:1 square</option>
                <option value="4:5">4:5 social</option>
              </select>
            </label>
            <label className={styles.field}>
              <span>Duration (s)</span>
              <input
                type="number"
                min={1}
                max={20}
                value={duration}
                onChange={(e) =>
                  setDuration(Math.max(1, Math.min(20, Number(e.target.value))))
                }
                className={styles.input}
              />
            </label>
            <label className={styles.field}>
              <span>Resolution</span>
              <select
                value={resolution}
                onChange={(e) =>
                  setResolution(e.target.value as typeof resolution)
                }
                className={styles.select}
              >
                <option value="720p">720p</option>
                <option value="1080p">1080p</option>
                <option value="4k">4K</option>
              </select>
            </label>
            <label className={styles.field}>
              <span>Seed (optional)</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="auto"
                className={styles.input}
              />
            </label>
          </div>

          <div className={styles.summaryRow}>
            <p className={styles.summaryText}>
              Estimated charge for this run:&nbsp;
              <span className={`mono ${styles.summaryAmount}`}>
                {formatUsdCents(projectedCost)}
              </span>
              &nbsp;
              <span className={styles.summaryMuted}>
                ({duration}s × {formatUsdCents(ratePerSecondCents)}/s)
              </span>
            </p>
            <Button
              type="submit"
              size="md"
              variant="primary"
              disabled={submitting}
            >
              {submitting ? "Submitting…" : "Queue generation"}
            </Button>
          </div>

          {error && <p className={styles.error}>{error}</p>}
        </form>

        <aside className={styles.queue}>
          <header className={styles.queueHead}>
            <h2 className={styles.queueTitle}>Recent queue</h2>
            <p className={styles.queueSub}>Auto-refreshing every 2 seconds</p>
          </header>
          {recent.length === 0 ? (
            <p className={styles.queueEmpty}>
              No queued or finished jobs yet. Submit your first prompt above.
            </p>
          ) : (
            <ul className={styles.queueList}>
              {recent.map((r) => (
                <li key={r.id} className={styles.queueItem}>
                  <div className={styles.queueMain}>
                    <p className={styles.queuePrompt}>
                      {r.prompt.length > 90
                        ? r.prompt.slice(0, 90) + "…"
                        : r.prompt}
                    </p>
                    <p className={styles.queueMeta}>
                      <span className="mono">{r.resolution}</span>
                      <span aria-hidden> · </span>
                      <span className="mono">{r.durationSeconds}s</span>
                      <span aria-hidden> · </span>
                      <span>{formatRelative(r.createdAt)}</span>
                    </p>
                  </div>
                  <span
                    className={styles.queueStatus}
                    data-status={r.status}
                  >
                    {r.status === "RUNNING" && (
                      <span className={styles.spinner} aria-hidden />
                    )}
                    {prettyStatus(r.status)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </>
  );
}

function prettyStatus(s: string): string {
  switch (s) {
    case "QUEUED":
      return "Queued";
    case "RUNNING":
      return "Rendering";
    case "SUCCEEDED":
      return "Ready";
    case "FAILED":
      return "Failed";
    case "CANCELED":
      return "Canceled";
    default:
      return s;
  }
}

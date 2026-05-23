/**
 * Ogenti inference client.
 *
 * The retrofit model is built and trained inside the parent monorepo at
 * `/ogenti`. When the inference HTTP server (FastAPI) is online, configure
 * `OGENTI_INFERENCE_URL` and swap `MockOgentiClient` for `HttpOgentiClient`
 * via `getOgentiClient()` below. No call sites need to change.
 *
 * The interface is intentionally minimal — submission, status polling, and
 * cancellation. The product layer (queue management, billing, webhooks) lives
 * one level up.
 */

export type GenerationStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "canceled";

export interface GenerateRequest {
  prompt: string;
  negativePrompt?: string;
  aspectRatio: "16:9" | "9:16" | "1:1" | "4:5";
  durationSeconds: number; // 1..20
  resolution: "720p" | "1080p" | "4k";
  seed?: number;
  referenceUrl?: string;
  modelVersion?: string;
}

export interface GenerateResult {
  status: GenerationStatus;
  resultUrl?: string;
  thumbnailUrl?: string;
  progress: number; // 0..1
  errorMessage?: string;
  computeSeconds?: number;
}

export interface JobHandle {
  jobToken: string;
}

export interface OgentiClient {
  /** Submit a generation request. Returns a token used for subsequent polling. */
  submit(req: GenerateRequest): Promise<JobHandle>;

  /** Poll the status of a previously-submitted job. */
  status(jobToken: string): Promise<GenerateResult>;

  /** Best-effort cancellation. Idempotent. */
  cancel(jobToken: string): Promise<void>;
}

// ─────────────────────────────────────────────────────────────────────────
// Mock implementation — deterministic timeline for staging + local dev
// ─────────────────────────────────────────────────────────────────────────

const MOCK_RENDER_SECONDS = 8;

type MockJob = {
  startedAt: number;
  req: GenerateRequest;
  canceled: boolean;
};

const mockStore = new Map<string, MockJob>();

export class MockOgentiClient implements OgentiClient {
  async submit(req: GenerateRequest): Promise<JobHandle> {
    const jobToken = `mock_${Date.now().toString(36)}_${Math.random()
      .toString(36)
      .slice(2, 10)}`;
    mockStore.set(jobToken, {
      startedAt: Date.now(),
      req,
      canceled: false,
    });
    return { jobToken };
  }

  async status(jobToken: string): Promise<GenerateResult> {
    const job = mockStore.get(jobToken);
    if (!job) {
      return { status: "failed", progress: 0, errorMessage: "Job not found." };
    }
    if (job.canceled) {
      return { status: "canceled", progress: 0 };
    }
    const elapsed = (Date.now() - job.startedAt) / 1000;
    const progress = Math.min(1, elapsed / MOCK_RENDER_SECONDS);
    if (progress >= 1) {
      return {
        status: "succeeded",
        progress: 1,
        // Deterministic placeholder until the real renderer is plumbed in.
        // We do NOT ship real video files; UI handles a missing thumb gracefully.
        resultUrl: `https://placeholders.ogenti.dev/mock/${jobToken}.mp4`,
        thumbnailUrl: `https://placeholders.ogenti.dev/mock/${jobToken}.jpg`,
        computeSeconds: Math.round(elapsed),
      };
    }
    if (progress > 0.05) {
      return { status: "running", progress };
    }
    return { status: "queued", progress: 0 };
  }

  async cancel(jobToken: string): Promise<void> {
    const job = mockStore.get(jobToken);
    if (job) {
      job.canceled = true;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// HTTP implementation (placeholder — fill in once the inference server is up)
// ─────────────────────────────────────────────────────────────────────────

export class HttpOgentiClient implements OgentiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiToken: string,
  ) {}

  private async req<T>(path: string, init: RequestInit = {}): Promise<T> {
    const r = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.apiToken}`,
        ...(init.headers ?? {}),
      },
      cache: "no-store",
    });
    if (!r.ok) {
      throw new Error(`ogenti inference ${path} failed: ${r.status} ${r.statusText}`);
    }
    return (await r.json()) as T;
  }

  async submit(req: GenerateRequest): Promise<JobHandle> {
    return this.req<JobHandle>("/v1/generations", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async status(jobToken: string): Promise<GenerateResult> {
    return this.req<GenerateResult>(`/v1/generations/${jobToken}`);
  }

  async cancel(jobToken: string): Promise<void> {
    await this.req<void>(`/v1/generations/${jobToken}/cancel`, {
      method: "POST",
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────

let cached: OgentiClient | undefined;

export function getOgentiClient(): OgentiClient {
  if (cached) return cached;
  const url = process.env.OGENTI_INFERENCE_URL;
  const token = process.env.OGENTI_INFERENCE_TOKEN;
  if (url && token) {
    cached = new HttpOgentiClient(url, token);
  } else {
    cached = new MockOgentiClient();
  }
  return cached;
}

/**
 * On-disk HTTP cache for everything the offline renderer fetches from the network (sample packs, site sample maps, GM
 * soundfonts). Before it, every chunk opened a FRESH page and every render a fresh browser, so all samples were
 * downloaded again per chunk (a 150 s render = ~8 downloads of the whole pack; one slow R2 fetch stalled a chunk for minutes).
 *
 * Served to the page through puppeteer request interception (`request.respond`), keyed by URL. Chosen over a shared
 * `userDataDir` HTTP cache because that one is decided by Chromium (heuristic freshness from Last-Modified, no control over a
 * rebuilt pack, eviction by size, disabled for incognito/fresh contexts, locked by a concurrent browser on the same profile),
 * whereas this one has explicit, testable freshness and is safe for concurrent renders.
 *
 * Invalidation (the pack is re-uploaded under the SAME URLs when it is rebuilt): validators + a freshness window.
 *  - an entry younger than the TTL is served without any request;
 *  - an older one is revalidated with a conditional GET (If-None-Match / If-Modified-Since): 304 renews it, 200 replaces it
 *    (write-through), so a rebuilt pack is picked up by the first render after the TTL, at the price of one tiny request per URL;
 *  - an upstream that cannot be reached while an entry exists serves the entry (stale-if-error, logged): an outage is only
 *    waited out for a URL we have never fetched;
 *  - an upstream without validators (no ETag / Last-Modified) is refetched in full after the TTL.
 * The TTL is MIDIGREP_RENDER_SAMPLE_CACHE_TTL_S (default 600 s: longer than any single render run's reuse window, shorter than
 * a pack rebuild + upload cycle that precedes a render; 0 = revalidate every URL once per process).
 * The body is stored under its sha256 and the meta file naming it is renamed into place last, so a reader never sees a torn
 * entry and concurrent renders race harmlessly (identical content, atomic renames).
 */
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

export interface CacheMeta { url: string; status: number; contentType: string; etag: string; lastModified: string; fetchedAt: number; bytes: number; sha256: string }
export interface CacheResult { status: number; contentType: string; body: Buffer; from: 'cache' | 'revalidated' | 'network' | 'stale' }
export interface CacheStats { cache: number; revalidated: number; network: number; stale: number; requests: number }
/** A network failure on a URL with no cached copy. `code` is the Chromium net error to hand back to the page. */
export class SampleFetchError extends Error {
  constructor(message: string, public code: 'timedout' | 'namenotresolved' | 'connectionrefused' | 'connectionreset' | 'internetdisconnected') { super(message); }
}
export interface RawResponse { status: number; contentType: string; etag: string; lastModified: string; body: Buffer }
export type RawFetch = (url: string, headers: Record<string, string>, stallMs: () => number, onActivity: () => void) => Promise<RawResponse>;

export function errorCode(err: any): SampleFetchError['code'] {
  const t = `${err?.code ?? ''} ${err?.cause?.code ?? ''} ${err?.name ?? ''} ${err?.message ?? ''}`;
  if (/stall|abort|timeout|ETIMEDOUT/i.test(t)) return 'timedout';
  if (/ENOTFOUND|EAI_AGAIN/.test(t)) return 'namenotresolved';
  if (/ECONNREFUSED/.test(t)) return 'connectionrefused';
  return 'connectionreset';
}

/** GET with a stall guard: aborts when no byte (or response) arrives for stallMs(). A slow transfer that keeps delivering is never cut. */
export const nodeFetch: RawFetch = async (url, headers, stallMs, onActivity) => {
  const ac = new AbortController();
  let timer: NodeJS.Timeout | undefined; let stalled = false;
  const arm = () => { clearTimeout(timer); timer = setTimeout(() => { stalled = true; ac.abort(); }, stallMs()); };
  try {
    arm();
    const res = await fetch(url, { headers, signal: ac.signal, redirect: 'follow' });
    onActivity(); arm();
    const parts: Buffer[] = [];
    if (res.body) for await (const c of res.body as any) { parts.push(Buffer.from(c)); onActivity(); arm(); }
    return { status: res.status, contentType: res.headers.get('content-type') ?? '', etag: res.headers.get('etag') ?? '', lastModified: res.headers.get('last-modified') ?? '', body: Buffer.concat(parts) };
  } catch (err: any) {
    if (stalled) throw new SampleFetchError(`no data for ${stallMs() / 1000} s from ${url}`, 'timedout');
    throw new SampleFetchError(`${err?.cause?.code ?? ''} ${err?.message ?? err}`.trim(), errorCode(err));
  } finally { clearTimeout(timer); }
};

export class SampleCache {
  stats: CacheStats = { cache: 0, revalidated: 0, network: 0, stale: 0, requests: 0 };
  private inflight = new Map<string, Promise<CacheResult>>();
  private checked = new Map<string, number>(); // url -> when this process last validated it (TTL 0 = once per process; also stops a retry storm during an outage)
  constructor(private o: { dir: string; ttlS: number; stallMs: () => number; onActivity?: () => void; log?: (s: string) => void; fetchRaw?: RawFetch; now?: () => number }) {
    fs.mkdirSync(o.dir, { recursive: true });
  }
  private now() { return (this.o.now ?? Date.now)(); }
  private key(url: string) { return crypto.createHash('sha256').update(url).digest('hex'); }
  private metaPath(url: string) { return path.join(this.o.dir, this.key(url) + '.json'); }
  private bodyPath(sha: string) { return path.join(this.o.dir, sha + '.body'); }
  private atomicWrite(file: string, data: Buffer | string) {
    const tmp = `${file}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`;
    fs.writeFileSync(tmp, data); fs.renameSync(tmp, file);
  }
  read(url: string): { meta: CacheMeta; body: Buffer } | null {
    try {
      const meta: CacheMeta = JSON.parse(fs.readFileSync(this.metaPath(url), 'utf8'));
      if (meta.url !== url) return null;
      const body = fs.readFileSync(this.bodyPath(meta.sha256));
      return body.length === meta.bytes ? { meta, body } : null;
    } catch { return null; }
  }
  private store(url: string, r: RawResponse): CacheMeta {
    const sha256 = crypto.createHash('sha256').update(r.body).digest('hex');
    const bp = this.bodyPath(sha256);
    if (!fs.existsSync(bp)) this.atomicWrite(bp, r.body);          // body first (immutable, named by content)
    const meta: CacheMeta = { url, status: r.status, contentType: r.contentType, etag: r.etag, lastModified: r.lastModified, fetchedAt: this.now(), bytes: r.body.length, sha256 };
    this.atomicWrite(this.metaPath(url), JSON.stringify(meta));    // meta last = the commit
    return meta;
  }
  get(url: string): Promise<CacheResult> {
    this.stats.requests++;
    let p = this.inflight.get(url);
    if (!p) { p = this.resolve(url).finally(() => this.inflight.delete(url)); this.inflight.set(url, p); }
    return p;
  }
  private async resolve(url: string): Promise<CacheResult> {
    const raw = this.o.fetchRaw ?? nodeFetch, act = this.o.onActivity ?? (() => {});
    const hit = this.read(url);
    const fresh = (t: number) => this.now() - t < this.o.ttlS * 1000;
    // served without a request: within the TTL, or already validated (or found unreachable) by this process
    if (hit && (fresh(hit.meta.fetchedAt) || this.checked.has(url))) { this.stats.cache++; return { status: hit.meta.status, contentType: hit.meta.contentType, body: hit.body, from: 'cache' }; }
    const cond: Record<string, string> = {};
    if (hit?.meta.etag) cond['if-none-match'] = hit.meta.etag;
    else if (hit?.meta.lastModified) cond['if-modified-since'] = hit.meta.lastModified;
    let r: RawResponse;
    try { r = await raw(url, cond, this.o.stallMs, act); }
    catch (err: any) {
      if (hit) { this.checked.set(url, this.now()); this.stats.stale++; this.o.log?.(`[SAMPLE-CACHE] ${url.slice(0, 120)}: upstream unreachable (${String(err.message).slice(0, 80)}), serving the cached copy`); return { status: hit.meta.status, contentType: hit.meta.contentType, body: hit.body, from: 'stale' }; }
      throw err;
    }
    this.checked.set(url, this.now());
    if (r.status === 304 && hit) {
      this.stats.revalidated++;
      this.atomicWrite(this.metaPath(url), JSON.stringify({ ...hit.meta, fetchedAt: this.now() }));
      return { status: hit.meta.status, contentType: hit.meta.contentType, body: hit.body, from: 'revalidated' };
    }
    if (r.status >= 500 && hit) { this.checked.set(url, this.now()); this.stats.stale++; this.o.log?.(`[SAMPLE-CACHE] ${url.slice(0, 120)}: upstream HTTP ${r.status}, serving the cached copy`); return { status: hit.meta.status, contentType: hit.meta.contentType, body: hit.body, from: 'stale' }; }
    this.stats.network++;
    if (r.status === 200) this.store(url, r);   // only a complete 200 is cached; 404/5xx pass through so a missing sound still fails loudly
    return { status: r.status, contentType: r.contentType, body: r.body, from: 'network' };
  }
}

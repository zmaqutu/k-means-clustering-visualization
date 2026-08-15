import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://clustering.example/", {
      headers: { accept: "text/html", host: "clustering.example", "x-forwarded-proto": "https" },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the clustering lab shell and share metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Clustering 3D Lab<\/title>/i);
  assert.match(html, /interactive Three\.js laboratory/i);
  assert.match(html, /property="og:image"/i);
  assert.match(html, /content="https:\/\/clustering\.example\/og-clustering-3d-lab\.png"/i);
  assert.match(html, /name="twitter:card"/i);
  assert.match(html, /content="summary_large_image"/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

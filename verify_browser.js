// Browser verification for eurosat-decision-tool viewer
// Launches headless Chrome, loads the tool page, checks viewer grid,
// captures console errors + network requests, reports results.

const CDP = require("chrome-remote-interface");
const { spawn } = require("child_process");
const os = require("os");

const PORT = 8765;
const URL = `http://localhost:${PORT}/`;
const DEBUG_PORT = 9223;

async function main() {
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (e) {}

  spawn(
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    [
      "--headless=new",
      "--disable-gpu",
      `--remote-debugging-port=${DEBUG_PORT}`,
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${os.tmpdir()}/chrome_profile_verify2`,
    ],
    { stdio: "ignore", detached: true }
  ).unref();
  await new Promise((r) => setTimeout(r, 5000));

  const client = await CDP({ port: DEBUG_PORT });
  const { Page, Runtime, Network } = client;

  await Promise.all([Page.enable(), Runtime.enable(), Network.enable()]);

  const consoleErrors = [];
  const pageExceptions = [];
  const networkRequests = [];

  Runtime.on("consoleAPICalled", (payload) => {
    const args = (payload.args || []).map((a) => a.value || a.description || String(a));
    const entry = { type: payload.type, args, timestamp: payload.timestamp };
    if (payload.type === "error" || payload.type === "exception") consoleErrors.push(entry);
  });
  Runtime.on("exceptionThrown", (payload) => {
    pageExceptions.push(payload.exceptionDetails ? payload.exceptionDetails.text : String(payload));
  });
  Network.on("requestWillBeSent", (payload) => {
    networkRequests.push({ url: payload.request.url, type: payload.type });
  });

  await Page.navigate({ url: URL });
  await Page.loadEventFired();
  await new Promise((r) => setTimeout(r, 30000)); // allow model load + viewer build

  const evalRes = await Runtime.evaluate({
    expression: `
      (() => {
        const cards = document.querySelectorAll('.result-card');
        const summary = document.getElementById('summary')?.textContent || "";
        const note = document.getElementById('viewerNote')?.textContent || "";
        const loadStatus = document.getElementById('loadStatus')?.textContent || "";
        const results = [];
        cards.forEach((c) => {
          const img = c.querySelector('.thumb');
          const trueLabel = c.querySelector('.true-label')?.textContent || "";
          const predLabel = c.querySelector('.pred-label')?.textContent || "";
          const prob = c.querySelector('.prob')?.textContent || "";
          const decision = c.querySelector('.decision')?.textContent || "";
          const badge = c.querySelector('.badge')?.textContent || "";
          results.push({
            src: img ? img.src : "",
            alt: img ? img.alt : "",
            trueLabel, predLabel, prob, decision, badge,
            isMatch: c.classList.contains('match'),
            isMismatch: c.classList.contains('mismatch'),
            isAbstained: c.classList.contains('abstained'),
          });
        });
        return { cards: results, summary, note, loadStatus, cardCount: cards.length };
      })()
    `,
    returnByValue: true,
    timeout: 60000,
  });

  const raw = evalRes && evalRes.result;
  const data = raw && raw.value ? raw.value : {};

  console.log("=== VIEWER VERIFICATION ===");
  console.log("loadStatus:", data.loadStatus);
  console.log("cardCount:", data.cardCount);
  console.log("summary:", data.summary);
  console.log("note:", data.note);
  console.log("=== CARDS ===");
  let matches = 0, mismatches = 0, answered = 0, abstained = 0;
  for (const c of (data.cards || [])) {
    console.log(`  [${c.alt}] true=${c.trueLabel} pred=${c.predLabel} ${c.prob} decision=${c.decision} badge=${c.badge} match=${c.isMatch} abstained=${c.isAbstained}`);
    if (c.isMatch) matches++; else mismatches++;
    if (c.decision && c.decision.startsWith("Answered")) answered++;
    else if (c.decision && c.decision.startsWith("Abstained")) abstained++;
  }
  console.log(`Computed: matches=${matches} mismatches=${mismatches} answered=${answered} abstained=${abstained}`);

  const labelsCheck = await Runtime.evaluate({
    expression: `
      (() => {
        const t = document.body.textContent || "";
        return {
          hasOnlyUnbiased: t.includes("Only unbiased figure"),
          hasValidationSelected: t.includes("validation-selected"),
          hasOverall09627: t.includes("0.9627"),
          hasVal09679: t.includes("0.9679"),
          hasThreshold050: t.includes("0.50"),
          hasEuroSATOnly: t.includes("EuroSAT only"),
          hasNotField: t.includes("not field imagery"),
          hasSeed42: t.includes("seed 42"),
          hasNoCI: t.includes("no confidence interval"),
          hasScopeNote: t.includes("Scope note"),
          hasSummaryNotAccuracy: t.includes("not a test-set accuracy"),
          hasConfusionMatrixNote: t.includes("not available as a static PNG"),
        };
      })()
    `,
    returnByValue: true,
  });
  console.log("=== LABEL CHECKS ===");
  console.log(JSON.stringify((labelsCheck && labelsCheck.result && labelsCheck.result.value) || labelsCheck, null, 2));

  console.log("=== CONSOLE ERRORS ===");
  console.log(consoleErrors.length === 0 ? "(none)" : JSON.stringify(consoleErrors, null, 2));

  console.log("=== PAGE EXCEPTIONS ===");
  console.log(pageExceptions.length === 0 ? "(none)" : JSON.stringify(pageExceptions, null, 2));

  const nonLocalhost = networkRequests.filter(
    (n) => !n.url.startsWith("http://localhost:") && !n.url.startsWith("http://127.0.0.1:") && !n.url.startsWith("http://[::1]:")
  );
  console.log("=== NETWORK ===");
  console.log(`Total requests: ${networkRequests.length}`);
  console.log(`Non-localhost requests: ${nonLocalhost.length}`);
  for (const n of nonLocalhost) console.log("  NON-LOCALHOST:", n.url);
  console.log("Sample requests:");
  for (const n of networkRequests.slice(0, 20)) console.log("  ", n.type, n.url);

  await client.close();
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (e) {}
  console.log("=== DONE ===");
}

const { execSync } = require("child_process");
main().catch((e) => {
  console.error("VERIFICATION FAILED:", e);
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (_) {}
  process.exit(1);
});

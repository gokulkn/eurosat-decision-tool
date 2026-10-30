// Browser verification for eurosat-decision-tool
// Launches headless Chrome, loads the tool page, runs all 12 examples,
// captures console errors + network requests, reports results.

const CDP = require("chrome-remote-interface");
const { spawn } = require("child_process");
const os = require("os");

const PORT = 8765;
const URL = `http://localhost:${PORT}/`;
const DEBUG_PORT = 9222;

async function main() {
  // Kill any existing chrome
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (e) {}

  // Launch Chrome headless with remote debugging (detached, non-blocking)
  spawn(
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    [
      "--headless=new",
      "--disable-gpu",
      `--remote-debugging-port=${DEBUG_PORT}`,
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${os.tmpdir()}/chrome_profile_verify`,
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
  await new Promise((r) => setTimeout(r, 25000)); // allow model load + initial inference

  const evalRes = await Runtime.evaluate({
    expression: `
      (async () => {
        const results = [];
        const thumbs = document.querySelectorAll('.thumb');
        for (let i = 0; i < thumbs.length; i++) {
          const t = thumbs[i];
          t.click();
          await new Promise(res => setTimeout(res, 3500));
          const topClass = document.getElementById('topClass').textContent;
          const topConf = document.getElementById('topConf').textContent;
          const decision = document.getElementById('decision').textContent;
          const status = document.getElementById('status').textContent;
          results.push({ index: i, alt: t.alt, title: t.title, topClass, topConf, decision, status });
        }
        return results;
      })()
    `,
    returnByValue: true,
    awaitPromise: true,
    timeout: 300000,
  });

  const raw = evalRes && evalRes.result;
  let arr = [];
  if (raw && raw.value) arr = raw.value;

  console.log("=== EXAMPLE RESULTS ===");
  let answered = 0, abstained = 0;
  for (const r of arr) {
    console.log(`#${r.index} [${r.alt}] topClass=${r.topClass} topConf=${r.topConf}`);
    console.log(`   title: ${r.title}`);
    console.log(`   decision: ${r.decision}`);
    console.log(`   status: ${r.status}`);
    if (r.decision && r.decision.startsWith("Answered")) answered++;
    else if (r.decision && r.decision.startsWith("Abstained")) abstained++;
  }
  console.log(`Total examples run: ${arr.length}`);
  console.log(`Answered: ${answered}, Abstained: ${abstained}`);

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
  for (const n of networkRequests.slice(0, 15)) console.log("  ", n.type, n.url);

  await client.close();
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (e) {}
  console.log("=== DONE ===");
}

main().catch((e) => {
  console.error("VERIFICATION FAILED:", e);
  try { execSync("taskkill /F /IM chrome.exe /T", { stdio: "ignore" }); } catch (_) {}
  process.exit(1);
});

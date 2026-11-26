// Records a walkthrough of the Promptly web app with Playwright.
//
//   PROMPTLY_HOME=$(mktemp -d) promptly serve --port 8790 &
//   node demo/record_demo.js http://127.0.0.1:8790 demo/out
//
// Everything shown is the real app responding live; the only thing this script adds is the
// caption bar at the bottom of the screen.
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://127.0.0.1:8790";
const OUT = process.argv[3] || "demo/out";
const W = 1280, H = 800;

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: W, height: H },
    deviceScaleFactor: 1,
    recordVideo: { dir: OUT, size: { width: W, height: H } },
    colorScheme: "light",
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  const pause = (ms) => page.waitForTimeout(ms);

  async function caption(text, ms = 2600) {
    await page.evaluate((t) => {
      let el = document.getElementById("__cap");
      if (!el) {
        el = document.createElement("div");
        el.id = "__cap";
        Object.assign(el.style, {
          position: "fixed", left: "50%", bottom: "22px", transform: "translateX(-50%)",
          maxWidth: "980px", padding: "12px 20px", borderRadius: "12px", zIndex: 9999,
          background: "rgba(20,20,24,.92)", color: "#fff", font: "500 17px/1.4 system-ui, sans-serif",
          boxShadow: "0 8px 30px rgba(0,0,0,.25)", textAlign: "center", transition: "opacity .25s",
        });
        document.body.appendChild(el);
      }
      el.style.opacity = t ? "1" : "0";
      if (t) el.innerHTML = t;
    }, text);
    if (ms) await pause(ms);
  }

  async function spot(selector) {
    // Brief highlight ring so viewers can follow the clicks.
    await page.$eval(selector, (el) => {
      el.scrollIntoView({ block: "center", behavior: "instant" });
      const r = el.getBoundingClientRect();
      const ring = document.createElement("div");
      Object.assign(ring.style, {
        position: "fixed", left: r.left - 6 + "px", top: r.top - 6 + "px", width: r.width + 12 + "px",
        height: r.height + 12 + "px", border: "3px solid #f59f00", borderRadius: "10px", zIndex: 9998,
        pointerEvents: "none", transition: "opacity .6s",
      });
      document.body.appendChild(ring);
      setTimeout(() => (ring.style.opacity = "0"), 900);
      setTimeout(() => ring.remove(), 1600);
    });
    await pause(450);
  }

  async function click(selector) { await spot(selector); await page.click(selector); }
  async function type(selector, text, delay = 32) {
    await spot(selector);
    await page.fill(selector, "");
    await page.type(selector, text, { delay });
  }
  async function rewrite() {
    await click("#go");
    await page.waitForFunction(() => document.getElementById("go").textContent === "Rewrite");
    await pause(500);
  }
  async function scrollTo(selector) {
    await page.$eval(selector, (el) => el.scrollIntoView({ block: "start", behavior: "smooth" }));
    await pause(700);
  }

  await page.goto(BASE);
  await pause(800);
  await caption("<b>Promptly</b>: type a prompt loosely; get back what you meant, with your own context folded in.", 3200);
  await caption("This recording: real local neural embeddings (multilingual MiniLM) + the <i>offline</i> rewrite engine. With an API key, Claude writes the rewrites.", 4200);

  // 1. Learn facts from a bio
  await caption("1 · Teach it about you. Paste a bio and it proposes facts. Nothing is saved until you accept.", 0);
  await type("#learn-text", "I'm a staff engineer building agent infrastructure. I mostly use Python and Go. I prefer concise answers with code first.", 18);
  await click("#learn-btn");
  await page.waitForSelector("#accept");
  await pause(1600);
  await click("#accept");
  await pause(900);

  // 2. Add facts by hand, one sensitive
  await caption("Add facts by hand too. Each one has a sensitivity level: public, personal or <b>sensitive</b>.", 0);
  await type("#fact-text", "Vegetarian, no fish", 30);
  await page.selectOption("#fact-cat", "constraint");
  await page.selectOption("#fact-sens", "sensitive");
  await click("#add-fact button");
  await pause(500);
  await type("#fact-text", "Has a dog named Rex", 30);
  await page.selectOption("#fact-cat", "other");
  await page.selectOption("#fact-sens", "personal");
  await click("#add-fact button");
  await pause(1400);

  // 3. A coding prompt
  await caption("2 · A loose coding request…", 0);
  await type("#prompt", "write a python script to retry flaky api calls");
  await rewrite();
  await caption("It picks only the facts that change the answer, ranked by vector similarity (the score next to each one).", 3800);
  await scrollTo("#result");
  await caption("The dog and the diet were not relevant, so they stayed out. The sensitive fact never left the device.", 3800);

  // 4. Privacy gate + semantic match
  await scrollTo("header");
  await caption("3 · Privacy gate. Meal planning at the default sharing limit (personal):", 0);
  await type("#prompt", "plan my meals for the week");
  await rewrite();
  await scrollTo("#privacy");
  await caption("“Vegetarian” is sensitive, so it was withheld. It was not sent to the engine at all.", 3400);
  await scrollTo("header");
  await caption("Raise the limit for this one request…", 0);
  await spot("#sens");
  await page.selectOption("#sens", "sensitive");
  await pause(500);
  await rewrite();
  await scrollTo("#injected");
  await caption("Now it is used. Note “meals” → “vegetarian” share no keywords; the embedding made the match.", 4000);

  // 5. Cross-language
  await scrollTo("header");
  await caption("4 · The vectors are multilingual. Same request in Chinese:", 0);
  await type("#prompt", "帮我规划一周的饭菜", 120);
  await rewrite();
  await scrollTo("#injected");
  await caption("The Chinese prompt still finds the English “Vegetarian” fact.", 3200);

  // 6. Triage + clarifying questions
  await scrollTo("header");
  await page.selectOption("#sens", "personal");
  await caption("5 · Triage first: does the prompt say <b>what</b>, <b>where</b>, and what <b>done</b> looks like?", 0);
  await type("#prompt", "fix it", 90);
  await rewrite();
  await caption("Too vague, so it names the gaps and asks once, only about what's missing.", 3000);
  await type("#q-list input >> nth=0", "make my retry script back off exponentially", 22);
  await type("#q-list input >> nth=1", "the python API client in my agent service", 22);
  await click("#again");
  await page.waitForFunction(() => document.getElementById("go").textContent === "Rewrite");
  await pause(600);
  await caption("Answers go back into the rewrite.", 2800);

  // 7. Feedback learning
  await scrollTo("header");
  await caption("6 · It learns from feedback. Another coding request:", 0);
  await type("#prompt", "write a python function that retries flaky http calls", 18);
  await rewrite();
  await scrollTo("#injected");
  await caption("Say I don't want the style preference used for this kind of request: 👎", 2200);
  await click('#injected .chip:has-text("concise") .vote[data-label="-1"]');
  await pause(900);
  await caption("The vote is stored next to this prompt's vector, so <i>similar</i> prompts inherit it.", 3000);
  await scrollTo("header");
  await caption("Re-run the earlier prompt from step 2 (it used the preference at 0.32):", 0);
  await type("#prompt", "write a python script to retry flaky api calls", 18);
  await rewrite();
  await scrollTo("#injected");
  await caption("Now the preference is ranked out for this similar prompt.", 3200);
  await scrollTo("header");
  await caption("An unrelated prompt is unaffected:", 0);
  await type("#prompt", "draft a concise reply to my manager", 22);
  await rewrite();
  await scrollTo("#injected");
  await caption("The preference still applies here, because the 👎 only counts for nearby prompts.", 3400);

  await scrollTo("header");
  await caption("<b>Promptly</b>: intent triage · personal context · privacy by construction · vectors that learn from you.", 3800);
  await caption("", 600);

  await ctx.close();
  await browser.close();
  if (errors.length) { console.error("page errors:", errors); process.exit(1); }
  console.log("ok");
})();

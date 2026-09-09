const test = require("node:test");
const assert = require("node:assert/strict");
const { PacedText } = require("../../src/agent_runtime/web/stream.js");

test("initial buffer, measured delivery, and bounded terminal catch-up", () => {
  const stream = new PacedText({ bufferMs: 1000, maxLagMs: 1800 });
  stream.append("a".repeat(80), 0, 0);
  stream.append("b".repeat(80), 400, 400);
  stream.tick(999);
  assert.equal(stream.text, "");
  stream.tick(1050);
  assert.ok(stream.text.length > 0 && stream.text.length < 160);
  stream.append("c".repeat(1000), 1200, 1200);
  stream.finish(1300);
  stream.tick(1551);
  assert.equal(stream.text, stream.received);
  assert.equal(stream.pending, false);
});

test("faster generation has a faster reveal rate", () => {
  const slow = new PacedText({ bufferMs: 100, maxLagMs: 2000 });
  const fast = new PacedText({ bufferMs: 100, maxLagMs: 2000 });
  for (let i = 0; i < 20; i++) {
    slow.append("slow", i * 50, i * 50);
    fast.append("f".repeat(40), i * 50, i * 50);
  }
  assert.ok(fast.rate > slow.rate * 5);
  slow.tick(1050);
  fast.tick(1050);
  assert.ok(fast.visible > slow.visible * 5);
});

test("no broken emoji, including graphemes split across input chunks", () => {
  const stream = new PacedText({ bufferMs: 20, maxLagMs: 100 });
  stream.append("🐍".repeat(40) + "👩", 0);
  stream.append("‍💻" + "नमस्ते", 5);
  for (let time = 20; time < 400; time += 7) {
    stream.tick(time);
    assert.ok(!/[\uD800-\uDBFF]$/.test(stream.text));
    assert.ok(!stream.text.endsWith("👩") && !stream.text.endsWith("👩‍"));
  }
  stream.finish(410, true);
  assert.equal(stream.text, "🐍".repeat(40) + "👩‍💻नमस्ते");
});

test("zero buffer, reduced motion, and explicit flush are immediate", () => {
  for (const options of [{ bufferMs: 0 }, { smooth: false }]) {
    const stream = new PacedText(options);
    stream.append("Immediate output", 0);
    assert.equal(stream.text, "Immediate output");
  }
  const stream = new PacedText();
  stream.append("An approval must not wait for animation.", 0);
  stream.flush();
  assert.equal(stream.text, stream.received);
});

test("a backgrounded tab catches up without a long replay", () => {
  const stream = new PacedText();
  stream.append("x".repeat(20000), 0);
  stream.finish(100);
  stream.tick(60000);
  assert.equal(stream.text.length, 20000);
});

test("an ongoing burst is shown within the configured maximum lag", () => {
  const stream = new PacedText({ bufferMs: 1000, maxLagMs: 1800 });
  stream.append("slow", 0, 0);
  stream.append("x".repeat(10000), 5000, 5000);
  for (let now = 5100; now <= 6850; now += 25) stream.tick(now);
  assert.equal(stream.visible, stream.received.length);
});

/* Small, testable presentation primitives. No model execution or protocol decisions here. */
((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.RelayStreaming = api;
})(globalThis, () => {
  "use strict";

  class PacedText {
    constructor({ bufferMs = 1000, maxLagMs = 1800, smooth = true } = {}) {
      this.bufferMs = Math.max(0, bufferMs);
      this.maxLagMs = Math.max(100, maxLagMs, this.bufferMs);
      this.smooth = smooth && this.bufferMs > 0;
      this.received = "";
      this.visible = 0;
      this.rate = 0;
      this.firstAt = null;
      this.lastAt = null;
      this.lastSourceAt = null;
      this.lastTick = null;
      this.finishedAt = null;
      this.credit = 0;
      this.boundaries = [0];
      this.arrivals = [];
      this.arrivalHead = 0;
      this.segmenter =
        typeof Intl.Segmenter === "function"
          ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
          : null;
    }

    append(text, now, sourceAt = now) {
      if (!text) return;
      const oldLength = this.received.length;
      this.received += text;
      this.arrivals.push({ end: this.received.length, at: now });
      if (this.firstAt === null) this.firstAt = this.lastTick = now;
      if (this.lastSourceAt !== null) {
        const elapsed = Math.max(20, sourceAt - this.lastSourceAt);
        const sample = Math.min(20000, (text.length * 1000) / elapsed);
        this.rate = this.rate ? this.rate * 0.7 + sample * 0.3 : sample;
      } else this.rate = Math.max(20, text.length * 4);
      this.lastAt = now;
      this.lastSourceAt = sourceAt;
      // Re-segment only the last grapheme + the new suffix: split emoji/combining
      // characters may span provider chunks, but must not be revealed as halves.
      const start = this.boundaries.length > 1 ? this.boundaries.at(-2) : 0;
      while (this.boundaries.at(-1) > start) this.boundaries.pop();
      const tail = this.received.slice(start);
      if (this.segmenter) {
        for (const part of this.segmenter.segment(tail))
          this.boundaries.push(start + part.index + part.segment.length);
      } else {
        let index = start;
        for (const point of tail) {
          index += point.length;
          this.boundaries.push(index);
        }
      }
      if (!this.smooth) this.visible = this.received.length;
      if (this.finishedAt !== null && oldLength === this.visible)
        this.visible = this.received.length;
    }

    finish(now, immediate = false) {
      this.finishedAt = now;
      if (immediate || !this.smooth || this.received.length < 240)
        this.visible = this.received.length;
    }

    flush() {
      this.visible = this.received.length;
    }

    tick(now) {
      const previous = this.visible;
      if (!this.pending) {
        this.lastTick = now;
        return false;
      }
      if (!this.smooth) this.flush();
      else if (this.finishedAt !== null && now - this.finishedAt >= 250)
        this.flush();
      else if (
        this.finishedAt !== null ||
        now - this.firstAt >= this.bufferMs
      ) {
        const elapsed = Math.max(0, Math.min(100, now - this.lastTick)) / 1000;
        const remaining = this.received.length - this.visible;
        // Follow measured delivery rate, but accelerate bursts so backlog cannot
        // grow indefinitely. Completed output drains within a bounded 250 ms.
        const catchup =
          this.finishedAt === null
            ? this.maxLagMs / 1000
            : Math.max(0.016, (250 - (now - this.finishedAt)) / 1000);
        while (
          this.arrivalHead < this.arrivals.length &&
          this.arrivals[this.arrivalHead].end <= this.visible
        )
          this.arrivalHead++;
        const oldest = this.arrivals[this.arrivalHead];
        const deadline = oldest
          ? Math.max(0.016, (oldest.at + this.maxLagMs - now) / 1000)
          : catchup;
        const dueRate = oldest ? (oldest.end - this.visible) / deadline : 0;
        const speed = Math.max(this.rate, remaining / catchup, dueRate, 1);
        this.credit += elapsed * speed;
        let target = Math.min(
          this.received.length,
          this.visible + Math.floor(this.credit),
        );
        for (
          let i = this.arrivalHead;
          i < this.arrivals.length &&
          now - this.arrivals[i].at >= this.maxLagMs;
          i++
        )
          target = Math.max(target, this.arrivals[i].end);
        if (this.arrivalHead > 100) {
          this.arrivals = this.arrivals.slice(this.arrivalHead);
          this.arrivalHead = 0;
        }
        let lo = 0,
          hi = this.boundaries.length - 1;
        while (lo < hi) {
          const mid = Math.ceil((lo + hi) / 2);
          if (this.boundaries[mid] <= target) lo = mid;
          else hi = mid - 1;
        }
        this.visible = Math.max(this.visible, this.boundaries[lo]);
        this.credit = Math.max(0, this.credit - (this.visible - previous));
      }
      this.lastTick = now;
      return this.visible !== previous;
    }

    get pending() {
      return this.visible < this.received.length;
    }
    get text() {
      return this.received.slice(0, this.visible);
    }
  }

  function key(node) {
    return node.nodeType === 1
      ? node.getAttribute("data-render-key") ||
          node.getAttribute("data-event-key") ||
          node.id ||
          null
      : null;
  }
  function compatible(a, b) {
    return a && a.nodeType === b.nodeType && a.nodeName === b.nodeName;
  }
  function patchNode(current, next) {
    if (current.nodeType === 3) {
      if (current.data !== next.data) {
        if (next.data.startsWith(current.data))
          current.appendData(next.data.slice(current.data.length));
        else current.replaceData(0, current.data.length, next.data);
      }
      return;
    }
    if (current.nodeType !== 1) return;
    // Native disclosure state and user drag-resizing belong to the viewer, not
    // to a newly rendered template. Keeping nodes also preserves focus/selection.
    const preserved = new Set(["style"]);
    if (current.tagName === "DETAILS") preserved.add("open");
    for (const attribute of [...current.attributes])
      if (!preserved.has(attribute.name) && !next.hasAttribute(attribute.name))
        current.removeAttribute(attribute.name);
    for (const attribute of [...next.attributes])
      if (
        !preserved.has(attribute.name) &&
        current.getAttribute(attribute.name) !== attribute.value
      )
        current.setAttribute(attribute.name, attribute.value);
    patchChildren(current, next);
  }
  function patchChildren(parent, next) {
    const children = [...parent.childNodes];
    const byKey = new Map(
      children.filter(key).map((node) => [key(node), node]),
    );
    const used = new Set();
    let cursor = parent.firstChild;
    for (const wanted of [...next.childNodes]) {
      const identity = key(wanted);
      let current = identity ? byKey.get(identity) : cursor;
      if (
        !compatible(current, wanted) ||
        used.has(current) ||
        (!identity && key(current))
      )
        current = null;
      if (!current) current = wanted.cloneNode(true);
      else patchNode(current, wanted);
      if (current !== cursor) parent.insertBefore(current, cursor);
      used.add(current);
      cursor = current.nextSibling;
    }
    for (const child of [...parent.childNodes])
      if (!used.has(child)) child.remove();
  }
  function reconcile(parent, html) {
    const template = document.createElement("template");
    template.innerHTML = html;
    patchChildren(parent, template.content);
  }
  return { PacedText, reconcile };
});

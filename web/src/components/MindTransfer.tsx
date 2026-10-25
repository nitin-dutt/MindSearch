import { useEffect, useRef } from "react";

// "The document turns into pixels and flows into the mind": while `active`, small square pixels break off the document icon
// (data-mind-source) and travel along curved paths into the logo (data-mind-target), each leaving a short stepped trail of fading pixels.
// Every arrival sends a ring of pixels out from the logo, and when `active` ends a last burst flies in. Everything is hard-edged and snapped
// to a pixel grid (no blur, no glow, no gradient) and is drawn on one canvas above the page, so nothing in the page moves.

type RGB = [number, number, number];
interface Particle {
  t: number; // 0 → 1 along its path
  dur: number; // seconds the flight takes
  sx: number; sy: number; // start
  cx: number; cy: number; // the point that bends the path
  jx: number; jy: number; // where it lands, relative to the logo centre
  size: number; // in grid cells
  tone: number; // 0 → 1: how far between the two pastel colours it is
}
interface Ripple {
  age: number;
}

const CELL = 3; // the pixel grid, in CSS pixels
const RATE = 20; // pixels per second while uploading
const START_BURST = 8;
const END_BURST = 18;
const TAIL = 4; // pixels in a trail
const RIPPLE_SECONDS = 0.8;

const FALLBACK_FROM: RGB = [6, 182, 212];
const FALLBACK_TO: RGB = [34, 197, 94];

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const rgba = (c: RGB, a: number) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a.toFixed(3)})`;
const snap = (v: number) => Math.round(v / CELL) * CELL;

function readColour(name: string, fallback: RGB): RGB {
  const m = getComputedStyle(document.documentElement).getPropertyValue(name).trim().match(/^#([0-9a-f]{6})$/i);
  return m ? [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)] : fallback;
}

/** The mind logo that is actually on screen: the sidebar one on a wide screen, the top-bar one when the sidebar is hidden. */
function visibleTarget(): HTMLElement | null {
  return Array.from(document.querySelectorAll<HTMLElement>("[data-mind-target]")).find((el) => el.getBoundingClientRect().width > 0) ?? null;
}

const centre = (el: Element) => {
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
};

interface Runner {
  spawning: boolean;
  running: boolean;
  begin: () => void;
  burst: (n: number) => void;
}

export function MindTransfer({ active }: { active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const runner = useRef<Runner | null>(null);

  // Build the animation loop once.
  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext?.("2d");
    if (!el || !ctx || /jsdom/i.test(navigator.userAgent) || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;

    const particles: Particle[] = [];
    const ripples: Ripple[] = [];
    const state: Runner = { spawning: false, running: false, begin: () => {}, burst: () => {} };
    let raf = 0;
    let last = 0;
    let carry = 0;
    let pulsing: HTMLElement | null = null;
    let from: RGB = FALLBACK_FROM;
    let to: RGB = FALLBACK_TO;

    const spawn = (a: { x: number; y: number }, b: { x: number; y: number }) => {
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const bend = (Math.random() < 0.5 ? -1 : 1) * (30 + Math.random() * 140); // some go one side of the straight line, some the other
      particles.push({
        t: 0,
        dur: 1 + Math.random() * 0.8,
        sx: a.x + (Math.random() - 0.5) * 26,
        sy: a.y + (Math.random() - 0.5) * 26,
        cx: mx + (-dy / len) * bend,
        cy: my + (dx / len) * bend - 30,
        jx: (Math.random() - 0.5) * 14,
        jy: (Math.random() - 0.5) * 14,
        size: Math.random() < 0.55 ? 1 : Math.random() < 0.75 ? 2 : 3,
        tone: Math.random(),
      });
    };

    const at = (p: Particle, t: number, goal: { x: number; y: number }) => {
      const u = ease(Math.min(1, Math.max(0, t)));
      const k = 1 - u;
      return {
        x: k * k * p.sx + 2 * k * u * p.cx + u * u * (goal.x + p.jx),
        y: k * k * p.sy + 2 * k * u * p.cy + u * u * (goal.y + p.jy),
      };
    };

    const stop = () => {
      state.running = false;
      cancelAnimationFrame(raf);
      ctx.clearRect(0, 0, el.width, el.height);
      pulsing?.classList.remove("mind-pulse");
      pulsing = null;
    };

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const ratio = window.devicePixelRatio || 1;
      if (el.width !== Math.round(innerWidth * ratio) || el.height !== Math.round(innerHeight * ratio)) {
        el.width = Math.round(innerWidth * ratio);
        el.height = Math.round(innerHeight * ratio);
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, innerWidth, innerHeight);

      const source = document.querySelector("[data-mind-source]");
      const target = visibleTarget();
      if (target && target !== pulsing) {
        pulsing?.classList.remove("mind-pulse");
        pulsing = target;
      }
      if (state.spawning) pulsing?.classList.add("mind-pulse");
      else pulsing?.classList.remove("mind-pulse");

      if (source && target) {
        const a = centre(source);
        const b = centre(target);
        if (state.spawning) {
          carry += dt * RATE;
          for (; carry >= 1; carry -= 1) spawn(a, b);
        }
        for (let i = particles.length - 1; i >= 0; i--) {
          const p = particles[i];
          p.t += dt / p.dur;
          if (p.t >= 1) {
            particles.splice(i, 1);
            if (ripples.length < 5) ripples.push({ age: 0 });
            continue;
          }
          const fade = Math.min(1, p.t * 8) * (p.t > 0.9 ? 1 - (p.t - 0.9) * 6 : 1);
          const colour = mix(from, to, p.tone);
          // the trail: older positions, smaller and fainter, each a square on the grid
          for (let s = TAIL; s >= 1; s--) {
            const q = at(p, p.t - s * 0.022, b);
            const strength = 1 - s / (TAIL + 1);
            const cells = Math.max(1, Math.round(p.size * strength));
            ctx.fillStyle = rgba(colour, fade * strength * 0.8);
            ctx.fillRect(snap(q.x), snap(q.y), cells * CELL, cells * CELL);
          }
          const head = at(p, p.t, b);
          ctx.fillStyle = rgba(from, fade);
          ctx.fillRect(snap(head.x), snap(head.y), p.size * CELL, p.size * CELL);
        }
        // a ring of pixels leaves the logo each time pixels arrive
        for (let i = ripples.length - 1; i >= 0; i--) {
          const r = ripples[i];
          r.age += dt;
          if (r.age >= RIPPLE_SECONDS) {
            ripples.splice(i, 1);
            continue;
          }
          const k = r.age / RIPPLE_SECONDS;
          const radius = 12 + k * 32;
          ctx.fillStyle = rgba(mix(from, to, 0.5), 0.75 * (1 - k));
          for (let n = 0; n < 24; n++) {
            const angle = (n / 24) * Math.PI * 2;
            ctx.fillRect(snap(b.x + Math.cos(angle) * radius), snap(b.y + Math.sin(angle) * radius), CELL, CELL);
          }
        }
      }

      if (!state.spawning && particles.length === 0 && ripples.length === 0) return stop();
      raf = requestAnimationFrame(frame);
    };

    const begin = () => {
      if (state.running) return;
      from = readColour("--accent", FALLBACK_FROM);
      to = readColour("--brand", FALLBACK_TO);
      state.running = true;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };

    state.burst = (n) => {
      const source = document.querySelector("[data-mind-source]");
      const target = visibleTarget();
      if (!source || !target) return;
      begin();
      for (let i = 0; i < n; i++) spawn(centre(source), centre(target));
    };
    state.begin = begin;
    runner.current = state;
    return () => {
      runner.current = null;
      stop();
    };
  }, []);

  // Start with a burst and a steady stream; when the upload ends, send one last burst and let everything finish flying.
  useEffect(() => {
    const r = runner.current;
    if (!r) return;
    if (active) {
      r.spawning = true;
      r.begin();
      r.burst(START_BURST);
    } else if (r.spawning) {
      r.spawning = false;
      r.burst(END_BURST);
    }
  }, [active]);

  return <canvas ref={canvas} aria-hidden className="pointer-events-none fixed inset-0 z-50 size-full" />;
}

'use strict';
/* PropSimulator dashboard.
   No build step and no external libraries: the globe is drawn with the 2D
   canvas API and the charts are hand-built SVG, so the page works from a
   plain file server and in a Codespace with nothing installed. */

const R_EARTH = 6371.0088;
const rad = d => d * Math.PI / 180, deg = r => r * 180 / Math.PI;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, n = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : v.toFixed(n);

/* ------------------------------------------------------------------ state */

const state = {
  frequency_mhz: 14.20, power_w: 100, tx_gain: 3, launch_angle_deg: 12.0,
  distance_km: 7496, max_hops: 4, tx_height_m: 15,
  f107: 140, kp: 2.30, sunspot_number: 95,
  tx_lat: 40.71, tx_lon: -74.01, rx_lat: 55.67, rx_lon: 37.20,
  when: '2026-08-30T13:00',
  fof2_scale: 1.00, hmf2_offset_km: 0, mode: 'O', ground: 'average_ground',
  rx_gain: 3, bandwidth_hz: 2400, noise_figure_db: 6.0, required_snr_db: 10,
  antenna_type: 'horizontal_dipole', rx_height_m: 12,
};

const layers = { land: true, night: true, shells: true, coverage: true, fan: true };

/* Controls are declared, not hand-written, so a slider and its readout can
   never drift apart and every one is wired the same way. */
const GROUPS = [
  { title: 'Link', open: true, items: [
    { k: 'frequency_mhz', label: 'Frequency', min: 1.8, max: 30, step: 0.01, unit: 'MHz', dec: 2 },
    { k: 'power_w', label: 'Transmitter power', min: 1, max: 1500, step: 1, unit: 'W' },
    { k: 'tx_gain', label: 'Transmitter gain', min: -6, max: 20, step: 0.5, unit: 'dBi', dec: 1 },
    { k: 'launch_angle_deg', label: 'Launch angle', min: 1, max: 60, step: 0.5, unit: '° elev', dec: 1 },
    { k: 'distance_km', label: 'Receiver distance', min: 200, max: 16000, step: 10, unit: 'km', dec: 0, moves: true },
    { k: 'max_hops', label: 'Maximum hops', min: 1, max: 8, step: 1, unit: 'reflections', dec: 0 },
    { k: 'tx_height_m', label: 'Antenna height', min: 1, max: 120, step: 1, unit: 'm', dec: 0 },
    { bands: true },
  ]},
  { title: 'Solar activity', open: true, items: [
    { k: 'f107', label: 'F10.7 solar flux', min: 65, max: 300, step: 1, unit: 'sfu', dec: 0 },
    { k: 'kp', label: 'Planetary Kp', min: 0, max: 9, step: 0.1, unit: '', dec: 2 },
    { k: 'sunspot_number', label: 'Sunspot number', min: 0, max: 250, step: 1, unit: '', dec: 0 },
  ]},
  { title: 'Geometry and time', open: true, items: [
    { k: 'tx_lat', label: 'TX latitude', min: -85, max: 85, step: 0.01, unit: '°', dec: 2 },
    { k: 'tx_lon', label: 'TX longitude', min: -180, max: 180, step: 0.01, unit: '°', dec: 2 },
    { k: 'rx_lat', label: 'RX latitude', min: -85, max: 85, step: 0.01, unit: '°', dec: 2 },
    { k: 'rx_lon', label: 'RX longitude', min: -180, max: 180, step: 0.01, unit: '°', dec: 2 },
    { k: 'when', label: 'Simulation time (UTC)', datetime: true },
  ]},
  { title: 'Ionospheric profile', open: true, items: [
    { k: 'fof2_scale', label: 'foF2 scale', min: 0.5, max: 2, step: 0.01, unit: '×', dec: 2 },
    { k: 'hmf2_offset_km', label: 'hmF2 offset', min: -100, max: 100, step: 1, unit: 'km', dec: 0 },
    { k: 'mode', label: 'Propagation mode', seg: [['O', 'Ordinary'], ['X', 'Extraordinary']] },
    { k: 'ground', label: 'Ground (reflections)', select: [
      ['salt_water', 'Salt water'], ['wet_ground', 'Wet ground'],
      ['average_ground', 'Average ground'], ['dry_ground', 'Dry ground'],
      ['urban', 'Urban'], ['ice', 'Ice'],
    ]},
  ]},
  { title: 'Receiver', open: true, items: [
    { k: 'rx_gain', label: 'Receiver gain', min: -6, max: 20, step: 0.5, unit: 'dBi', dec: 1 },
    { k: 'bandwidth_hz', label: 'Bandwidth', min: 100, max: 12000, step: 50, unit: 'Hz', dec: 0 },
    { k: 'noise_figure_db', label: 'Noise figure', min: 0, max: 25, step: 0.5, unit: 'dB', dec: 1 },
    { k: 'required_snr_db', label: 'Required SNR', min: -6, max: 40, step: 1, unit: 'dB', dec: 0 },
  ]},
];

const BANDS = [['160 m',1.9],['80 m',3.65],['40 m',7.1],['30 m',10.125],['20 m',14.2],
               ['17 m',18.1],['15 m',21.2],['12 m',24.94],['10 m',28.5]];

const PRESETS = {
  'Quiet Sun':        { f107: 70,  sunspot_number: 8,   kp: 1.0 },
  'Normal':           { f107: 140, sunspot_number: 95,  kp: 2.3 },
  'Solar Maximum':    { f107: 250, sunspot_number: 210, kp: 3.0 },
  'Geomagnetic Storm':{ f107: 160, sunspot_number: 110, kp: 7.7 },
  'Midnight':         { when: '2026-08-30T04:00' },
  'Long-Distance Link':{ rx_lat: -33.87, rx_lon: 151.21, frequency_mhz: 21.2 },
};

/* ------------------------------------------------------- spherical helpers */

function greatCircleKm(a, b) {
  const dφ = rad(b[0] - a[0]), dλ = rad(b[1] - a[1]);
  const h = Math.sin(dφ / 2) ** 2 +
    Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(dλ / 2) ** 2;
  return 2 * R_EARTH * Math.asin(Math.sqrt(Math.min(1, h)));
}
function bearingDeg(a, b) {
  const dλ = rad(b[1] - a[1]);
  const y = Math.sin(dλ) * Math.cos(rad(b[0]));
  const x = Math.cos(rad(a[0])) * Math.sin(rad(b[0])) -
            Math.sin(rad(a[0])) * Math.cos(rad(b[0])) * Math.cos(dλ);
  return (deg(Math.atan2(y, x)) + 360) % 360;
}
function destination(origin, bearing, km) {
  const δ = km / R_EARTH, θ = rad(bearing), φ1 = rad(origin[0]), λ1 = rad(origin[1]);
  const φ2 = Math.asin(clamp(Math.sin(φ1) * Math.cos(δ) + Math.cos(φ1) * Math.sin(δ) * Math.cos(θ), -1, 1));
  const λ2 = λ1 + Math.atan2(Math.sin(θ) * Math.sin(δ) * Math.cos(φ1),
                             Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2));
  return [deg(φ2), ((deg(λ2) + 540) % 360) - 180];
}
function interpolate(a, b, t) {
  const d = greatCircleKm(a, b) / R_EARTH;
  if (d < 1e-9) return a;
  const s = Math.sin(d), p = Math.sin((1 - t) * d) / s, q = Math.sin(t * d) / s;
  const x = p * Math.cos(rad(a[0])) * Math.cos(rad(a[1])) + q * Math.cos(rad(b[0])) * Math.cos(rad(b[1]));
  const y = p * Math.cos(rad(a[0])) * Math.sin(rad(a[1])) + q * Math.cos(rad(b[0])) * Math.sin(rad(b[1]));
  const z = p * Math.sin(rad(a[0])) + q * Math.sin(rad(b[0]));
  return [deg(Math.atan2(z, Math.hypot(x, y))), deg(Math.atan2(y, x))];
}

/* ------------------------------------------------------------------- globe */

class Globe {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.lon0 = -20; this.lat0 = 30; this.fill = 0.78;
    this.land = null; this.mask = document.createElement('canvas');
    this.mask.width = this.mask.height = 190;
    this._bind();
  }
  _bind() {
    let dragging = false, px = 0, py = 0;
    this.canvas.addEventListener('pointerdown', e => {
      dragging = true; px = e.clientX; py = e.clientY;
      this.canvas.classList.add('drag'); this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', e => {
      if (!dragging) return;
      this.lon0 -= (e.clientX - px) * 0.32;
      this.lat0 = clamp(this.lat0 + (e.clientY - py) * 0.32, -85, 85);
      px = e.clientX; py = e.clientY; this.draw();
    });
    const stop = () => { dragging = false; this.canvas.classList.remove('drag'); };
    this.canvas.addEventListener('pointerup', stop);
    this.canvas.addEventListener('pointercancel', stop);
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.fill = clamp(this.fill * (e.deltaY > 0 ? 0.93 : 1.075), 0.4, 2.6);
      this.draw();
    }, { passive: false });
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, r.width * dpr);
    this.canvas.height = Math.max(1, r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
    this.cx = this.w / 2; this.cy = this.h / 2;
    this.R = Math.min(this.w, this.h) * 0.5 * this.fill;
  }
  /* Orthographic. A raised point stays visible past the geometric horizon,
     which is why the visibility test is on the projected radius and not
     simply on the sign of the cosine. */
  project(lat, lon, altKm = 0) {
    const φ = rad(lat), λ = rad(lon), φ0 = rad(this.lat0), λ0 = rad(this.lon0);
    const cosc = Math.sin(φ0) * Math.sin(φ) + Math.cos(φ0) * Math.cos(φ) * Math.cos(λ - λ0);
    const scale = 1 + altKm / R_EARTH;
    const k = this.R * scale;
    const x = this.cx + k * Math.cos(φ) * Math.sin(λ - λ0);
    const y = this.cy - k * (Math.cos(φ0) * Math.sin(φ) - Math.sin(φ0) * Math.cos(φ) * Math.cos(λ - λ0));
    const ρ2 = (scale * scale) * (1 - cosc * cosc);
    return { x, y, visible: cosc >= 0 || ρ2 >= 1 };
  }
  setLand(polys) {
    // Densify so the limb clip reads as a curve rather than as facets.
    this.land = polys.map(ring => {
      const out = [];
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i], b = ring[(i + 1) % ring.length];
        const steps = Math.max(1, Math.ceil(greatCircleKm(a, b) / 220));
        for (let s = 0; s < steps; s++) out.push(interpolate(a, b, s / steps));
      }
      return out;
    });
  }
  _runs(points, altKm = 0) {
    const runs = []; let run = [];
    for (const p of points) {
      const q = this.project(p[0], p[1], altKm);
      if (q.visible) run.push(q);
      else { if (run.length > 1) runs.push(run); run = []; }
    }
    if (run.length > 1) runs.push(run);
    return runs;
  }
  _viewNormal() {
    const φ0 = rad(this.lat0), λ0 = rad(this.lon0);
    return [Math.cos(φ0) * Math.cos(λ0), Math.cos(φ0) * Math.sin(λ0), Math.sin(φ0)];
  }
  /* Clip a filled ring to the visible hemisphere.
     Done in three dimensions, against the plane through the centre of the
     Earth perpendicular to the view, and only then projected. Two earlier
     attempts were wrong in instructive ways: dropping hidden vertices closes
     the shape with a chord straight across the disc, and pulling them onto
     the limb along their own direction inflates a continent on the far side
     into a blob covering half the globe. Sutherland-Hodgman against the
     plane puts the boundary exactly on the horizon, and the edges that lie
     along it are then subdivided so they follow the limb as an arc rather
     than cutting across it as a straight line. */
  _clipRing(ring) {
    const n = this._viewNormal();
    const toXYZ = ([lat, lon]) => {
      const φ = rad(lat), λ = rad(lon);
      return [Math.cos(φ) * Math.cos(λ), Math.cos(φ) * Math.sin(λ), Math.sin(φ)];
    };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const norm = v => { const m = Math.hypot(v[0], v[1], v[2]) || 1;
                        return [v[0] / m, v[1] / m, v[2] / m]; };
    const cross = (a, b, t) => norm([a[0] + (b[0] - a[0]) * t,
                                     a[1] + (b[1] - a[1]) * t,
                                     a[2] + (b[2] - a[2]) * t]);

    const pts = ring.map(toXYZ);
    const out = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const da = dot(a, n), db = dot(b, n);
      if (da >= 0) out.push({ v: a, edge: false });
      if ((da >= 0) !== (db >= 0)) {
        out.push({ v: cross(a, b, da / (da - db)), edge: true });
      }
    }
    if (out.length < 3) return [];

    // Follow the horizon between two points that both sit on it.
    const dense = [];
    for (let i = 0; i < out.length; i++) {
      const a = out[i], b = out[(i + 1) % out.length];
      dense.push(a.v);
      if (!a.edge || !b.edge) continue;
      const angle = Math.acos(clamp(dot(a.v, b.v), -1, 1));
      const steps = Math.max(1, Math.round(angle / 0.05));
      for (let s = 1; s < steps; s++) {
        const t = s / steps, ω = angle;
        if (ω < 1e-9) break;
        const f1 = Math.sin((1 - t) * ω) / Math.sin(ω), f2 = Math.sin(t * ω) / Math.sin(ω);
        dense.push([a.v[0] * f1 + b.v[0] * f2, a.v[1] * f1 + b.v[1] * f2,
                    a.v[2] * f1 + b.v[2] * f2]);
      }
    }
    return dense.map(v => this.project(deg(Math.asin(clamp(v[2], -1, 1))),
                                       deg(Math.atan2(v[1], v[0]))));
  }
  _stroke(points, altKm, style, width, dash) {
    const c = this.ctx;
    c.save(); c.strokeStyle = style; c.lineWidth = width;
    if (dash) c.setLineDash(dash);
    for (const run of this._runs(points, altKm)) {
      c.beginPath(); c.moveTo(run[0].x, run[0].y);
      for (let i = 1; i < run.length; i++) c.lineTo(run[i].x, run[i].y);
      c.stroke();
    }
    c.restore();
  }
  smallCircle(centre, radiusKm, steps = 90) {
    const pts = [];
    for (let i = 0; i <= steps; i++) pts.push(destination(centre, i * 360 / steps, radiusKm));
    return pts;
  }
  _nightMask(subsolar) {
    const n = this.mask.width, mc = this.mask.getContext('2d');
    const img = mc.createImageData(n, n);
    const φ0 = rad(this.lat0), λ0 = rad(this.lon0);
    const sφ = rad(subsolar[0]), sλ = rad(subsolar[1]);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) / n * 2 - 1, y = 1 - (j + 0.5) / n * 2;
        const ρ = Math.hypot(x, y);
        const o = (j * n + i) * 4;
        if (ρ > 1) { img.data[o + 3] = 0; continue; }
        const c = Math.asin(clamp(ρ, 0, 1));
        const lat = Math.asin(clamp(Math.cos(c) * Math.sin(φ0) + (ρ ? y * Math.sin(c) * Math.cos(φ0) / ρ : 0), -1, 1));
        const lon = λ0 + Math.atan2(x * Math.sin(c), ρ * Math.cos(c) * Math.cos(φ0) - y * Math.sin(c) * Math.sin(φ0));
        const cosz = Math.sin(lat) * Math.sin(sφ) + Math.cos(lat) * Math.cos(sφ) * Math.cos(lon - sλ);
        // A soft edge over a few degrees, which is roughly what civil
        // twilight looks like and avoids an aliased hard rim.
        const night = clamp((0.09 - cosz) / 0.18, 0, 1);
        img.data[o] = 4; img.data[o + 1] = 10; img.data[o + 2] = 20;
        img.data[o + 3] = Math.round(night * 176);
      }
    }
    mc.putImageData(img, 0, 0);
  }
  draw(data) {
    if (data !== undefined) this.data = data;
    this.resize();
    const c = this.ctx, d = this.data;
    c.clearRect(0, 0, this.w, this.h);

    // ocean disc
    c.save();
    c.beginPath(); c.arc(this.cx, this.cy, this.R, 0, Math.PI * 2); c.clip();
    c.fillStyle = layers.land ? '#a9e9ec' : '#0d1b26';
    c.fillRect(0, 0, this.w, this.h);

    if (layers.land && this.land) {
      c.fillStyle = '#eef5f7'; c.strokeStyle = 'rgba(96,150,162,.65)'; c.lineWidth = 0.8;
      for (const ring of this.land) {
        const path = this._clipRing(ring);
        if (path.length < 3) continue;
        c.beginPath(); c.moveTo(path[0].x, path[0].y);
        for (let i = 1; i < path.length; i++) c.lineTo(path[i].x, path[i].y);
        c.closePath(); c.fill();
      }
      for (const ring of this.land) {
        for (const run of this._runs(ring)) {
          c.beginPath(); c.moveTo(run[0].x, run[0].y);
          for (let i = 1; i < run.length; i++) c.lineTo(run[i].x, run[i].y);
          c.stroke();
        }
      }
    }

    // graticule on the surface
    c.strokeStyle = 'rgba(40,90,110,.30)'; c.lineWidth = 0.6;
    this._graticule(0, 'rgba(38,92,112,.34)', 0.6);

    if (layers.night && d && d.geometry) {
      this._nightMask(d.geometry.subsolar);
      c.imageSmoothingEnabled = true;
      c.drawImage(this.mask, this.cx - this.R, this.cy - this.R, this.R * 2, this.R * 2);
    }
    c.restore();

    // limb
    c.save();
    c.beginPath(); c.arc(this.cx, this.cy, this.R, 0, Math.PI * 2);
    c.strokeStyle = 'rgba(69,199,224,.55)'; c.lineWidth = 1.2; c.stroke();
    c.restore();

    if (layers.shells) {
      this._graticule(110, 'rgba(69,199,224,.10)', 0.6, 30);
      this._graticule(300, 'rgba(69,199,224,.17)', 0.7, 30);
      for (const alt of [110, 300]) {
        const rim = [];
        for (let a = 0; a <= 360; a += 3) {
          rim.push([Math.sin(rad(a)) * 89.9, a - 180]);
        }
        const c = this.ctx; c.save();
        c.beginPath();
        c.arc(this.cx, this.cy, this.R * (1 + alt / R_EARTH), 0, Math.PI * 2);
        c.strokeStyle = alt === 300 ? 'rgba(69,199,224,.30)' : 'rgba(69,199,224,.18)';
        c.lineWidth = 0.9; c.stroke(); c.restore();
      }
    }
    if (d) this._drawLink(d);
  }
  /* Meridians stop short of the poles and thin out on the shells. Drawn all
     the way to 88 degrees they converge into a bright fan at whichever pole
     is in view, and with three shells stacked that fan washed out a quarter
     of the globe. */
  _graticule(altKm, style, width, meridianStep = 15) {
    for (let lat = -75; lat <= 75; lat += 15) {
      const pts = [];
      for (let lon = -180; lon <= 180; lon += 4) pts.push([lat, lon]);
      this._stroke(pts, altKm, style, width);
    }
    for (let lon = -180; lon < 180; lon += meridianStep) {
      const pts = [];
      for (let lat = -78; lat <= 78; lat += 4) pts.push([lat, lon]);
      this._stroke(pts, altKm, style, width);
    }
  }
  _drawLink(d) {
    const g = d.geometry, c = this.ctx;
    const tx = g.tx, rx = g.rx;

    if (layers.coverage) {
      if (d.ray.max_range_km) {
        this._stroke(this.smallCircle(tx, d.ray.max_range_km), 0,
                     'rgba(239,161,78,.55)', 1, [5, 4]);
      }
      this._stroke(g.great_circle, 0, 'rgba(198,213,227,.6)', 1.1, [4, 4]);
    }

    const bearing = g.bearing_deg;
    const arcPoints = (arc, hopKm) => arc.map(([f, h]) => ({
      p: destination(tx, bearing, f * hopKm), h,
    }));
    const drawArc = (pts, style, width) => {
      c.save(); c.strokeStyle = style; c.lineWidth = width;
      let started = false;
      c.beginPath();
      for (const { p, h } of pts) {
        const q = this.project(p[0], p[1], h);
        if (!q.visible) { started = false; continue; }
        if (!started) { c.moveTo(q.x, q.y); started = true; } else c.lineTo(q.x, q.y);
      }
      c.stroke(); c.restore();
    };

    // Two different things, drawn differently on purpose.
    //
    // The link path is how the signal actually reaches the receiver: the
    // mode the budget was built from, hop by hop.
    if (d.link_path) {
      const lp = d.link_path;
      let travelled = 0;
      for (let i = 0; i < lp.hops; i++) {
        // Each hop has its own range and its own arc: the ionosphere over
        // the first hop of a terminator-crossing circuit is not the one
        // over the last, and the hops reach different distances because
        // of it.
        const span = lp.hop_ranges_km[i];
        const arc = lp.arcs[i];
        const from = destination(tx, bearing, travelled);
        travelled += span;
        drawArc(arc.map(([f, h]) => ({ p: destination(from, bearing, f * span), h })),
                'rgba(69,199,224,.92)', 2);
      }
    }
    // The chosen ray is one hop at the launch angle on the slider. It is
    // not the link: it is the ray you are pointing at, and where it lands
    // is the question the readout answers.
    const hop = d.ray.hop_range_km || (d.ray.max_range_km || 2000);
    if (layers.fan) {
      for (const f of d.ray.fan) {
        drawArc(arcPoints(f.arc, hop), 'rgba(150,190,215,.26)', 0.8);
      }
    }
    drawArc(arcPoints(d.ray.arc, hop),
            d.ray.state === 'reflected' ? 'rgba(255,255,255,.85)' : 'rgba(239,107,116,.85)', 1.6);
    if (layers.coverage && d.ray.hop_range_km) {
      this._marker(destination(tx, bearing, d.ray.hop_range_km), '#ffffff', 3);
    }

    if (layers.coverage) {
      this._marker(tx, '#45c7e0');
      this._marker(rx, '#54d18b');
    }
  }
  _marker(p, colour, size = 4.5) {
    const q = this.project(p[0], p[1]); if (!q.visible) return;
    const c = this.ctx; c.save();
    c.beginPath(); c.arc(q.x, q.y, size, 0, Math.PI * 2);
    c.fillStyle = colour; c.fill();
    c.strokeStyle = 'rgba(6,10,17,.9)'; c.lineWidth = 1.5; c.stroke();
    c.restore();
  }
  framePath(g) {
    const mid = interpolate(g.tx, g.rx, 0.5);
    this.lat0 = clamp(mid[0], -80, 80); this.lon0 = mid[1];
    // Size the globe so the path's chord spans about half the shorter side.
    // A fixed zoom cannot do that: a 500 km hop and a 15000 km path need
    // opposite ends of the range, and guessing from the distance alone
    // pushed long paths past the edge of the panel.
    const halfAngle = g.distance_km / R_EARTH / 2;
    const chord = Math.max(Math.sin(halfAngle), 0.08);
    this.fill = clamp(0.5 / chord, 0.55, 1.35);
    this.draw();
  }
}

/* ------------------------------------------------------------------ charts */

const NS = 'http://www.w3.org/2000/svg';
function el(name, attrs, text) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}
function axes(svg, W, H, pad, xTicks, yTicks, xOf, yOf, yLabel) {
  for (const t of yTicks) {
    svg.append(el('line', { x1: pad.l, y1: yOf(t), x2: W - pad.r, y2: yOf(t),
                            stroke: '#172431', 'stroke-width': 1 }));
    svg.append(el('text', { x: pad.l - 6, y: yOf(t) + 3.5, fill: '#5d7488',
                            'font-size': 9, 'text-anchor': 'end' }, String(t)));
  }
  for (const t of xTicks) {
    svg.append(el('text', { x: xOf(t), y: H - 4, fill: '#5d7488',
                            'font-size': 9, 'text-anchor': 'middle' },
                  t >= 1000 ? (t / 1000) + 'k' : String(t)));
  }
  svg.append(el('text', { x: pad.l, y: pad.t - 3, fill: '#5d7488', 'font-size': 9 }, yLabel));
}
function polyline(svg, pts, stroke, width, dash) {
  if (pts.length < 2) return;
  const a = { points: pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '),
              fill: 'none', stroke, 'stroke-width': width };
  if (dash) a['stroke-dasharray'] = dash;
  svg.append(el('polyline', a));
}
/* Split at gaps so a skip zone shows as a break in the line rather than as a
   straight segment across it. A gap is a distance no ray lands on, not a
   weak signal, and drawing through it would say the opposite. */
function runsOf(samples, valueOf) {
  const runs = []; let run = [];
  for (const s of samples) {
    const v = valueOf(s);
    if (v === null || v === undefined) { if (run.length) runs.push(run); run = []; }
    else run.push([s.distance_km, v]);
  }
  if (run.length) runs.push(run);
  return runs;
}

function drawCoverage(svg, data, target) {
  svg.textContent = '';
  const W = svg.clientWidth || 520, H = svg.clientHeight || 160;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  if (!data) return;
  const pad = { l: 34, r: 10, t: 12, b: 16 };
  const S = data.samples;
  const xs = S.map(s => s.distance_km);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  // The ground wave is plotted on the same axes but only where it is worth
  // seeing: a surface field 150 dB under the noise would otherwise drag the
  // vertical scale down until the skywave curve was a flat line.
  const groundFloor = -40;
  const ground = S.filter(s => s.ground_wave_field_dbuv_m !== null
                            && s.ground_wave_field_dbuv_m > groundFloor);
  const vals = S.filter(s => s.reached).flatMap(s => [s.field_strength_dbuv_m, s.snr_db])
    .concat(ground.map(s => s.ground_wave_field_dbuv_m));
  if (!vals.length) {
    // An empty chart looks like a failure to compute. It is not: at this
    // frequency nothing lands anywhere in range by either route, and saying
    // so is the result.
    svg.append(el('text', { x: W / 2, y: H / 2, fill: '#5d7488', 'font-size': 11,
                            'text-anchor': 'middle' },
                  `nothing reaches any distance at ${data.frequency_mhz.toFixed(2)} MHz`));
    return;
  }
  const ymin = Math.min(-20, Math.floor(Math.min(...vals) / 20) * 20);
  const ymax = Math.max(20, Math.ceil(Math.max(...vals) / 20) * 20);
  const xOf = v => pad.l + (v - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const yOf = v => pad.t + (ymax - v) / (ymax - ymin) * (H - pad.t - pad.b);

  const yTicks = []; const step = Math.max(20, Math.round((ymax - ymin) / 4 / 20) * 20);
  for (let v = Math.ceil(ymin / step) * step; v <= ymax; v += step) yTicks.push(v);
  const xTicks = []; for (let v = 2000; v <= xmax; v += 2000) xTicks.push(v);
  axes(svg, W, H, pad, xTicks, yTicks, xOf, yOf, 'dBµV/m · dB');

  // Distances no ray lands on.
  let gap = null;
  for (const s of S) {
    if (!s.reached && gap === null) gap = s.distance_km;
    else if (s.reached && gap !== null) {
      svg.append(el('rect', { x: xOf(gap), y: pad.t, width: Math.max(1, xOf(s.distance_km) - xOf(gap)),
                              height: H - pad.t - pad.b, fill: 'rgba(239,107,116,.10)' }));
      gap = null;
    }
  }
  if (gap !== null) svg.append(el('rect', { x: xOf(gap), y: pad.t, width: Math.max(1, xOf(xmax) - xOf(gap)),
                                            height: H - pad.t - pad.b, fill: 'rgba(239,107,116,.10)' }));

  const req = data.required_snr_db;
  if (req >= ymin && req <= ymax) {
    polyline(svg, [[xOf(xmin), yOf(req)], [xOf(xmax), yOf(req)]], '#efa14e', 1, '4 3');
  }
  for (const r of runsOf(S, s => s.field_strength_dbuv_m)) polyline(svg, r.map(([x, y]) => [xOf(x), yOf(y)]), '#45c7e0', 1.6);
  for (const r of runsOf(S, s => s.snr_db)) polyline(svg, r.map(([x, y]) => [xOf(x), yOf(y)]), '#54d18b', 1.6);
  // Drawn last and dashed: inside the skip zone it is the only field there
  // is, and outside it, watching it fall away under the skywave is the
  // point of having both on one pair of axes.
  for (const r of runsOf(S, s => (s.ground_wave_field_dbuv_m !== null
                                  && s.ground_wave_field_dbuv_m > groundFloor)
                                 ? s.ground_wave_field_dbuv_m : null)) {
    polyline(svg, r.map(([x, y]) => [xOf(x), yOf(y)]), '#b58ce0', 1.4, '5 3');
  }

  if (target >= xmin && target <= xmax) {
    polyline(svg, [[xOf(target), pad.t], [xOf(target), H - pad.b]], '#8aa0b5', 1, '3 3');
    svg.append(el('text', { x: xOf(target) + 4, y: pad.t + 8, fill: '#8aa0b5', 'font-size': 9 },
                  'target ' + Math.round(target) + ' km'));
  }
}

function drawBand(svg, data, target, frequency) {
  svg.textContent = '';
  const W = svg.clientWidth || 520, H = svg.clientHeight || 160;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  if (!data) return;
  const pad = { l: 30, r: 10, t: 12, b: 16 };
  const S = data.samples;
  const xs = S.map(s => s.distance_km);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const highs = S.map(s => s.muf_mhz).filter(v => v !== null);
  if (!highs.length) {
    svg.append(el('text', { x: W / 2, y: H / 2, fill: '#5d7488', 'font-size': 11,
                            'text-anchor': 'middle' },
                  'the ionosphere returns nothing at any distance'));
    return;
  }
  const ymax = Math.max(32, Math.ceil((highs.length ? Math.max(...highs) : 30) / 5) * 5);
  const xOf = v => pad.l + (v - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const yOf = v => pad.t + (ymax - v) / ymax * (H - pad.t - pad.b);

  const yTicks = []; for (let v = 0; v <= ymax; v += 10) yTicks.push(v);
  const xTicks = []; for (let v = 2000; v <= xmax; v += 2000) xTicks.push(v);
  axes(svg, W, H, pad, xTicks, yTicks, xOf, yOf, 'MHz');

  // The window between LOF and MUF: where a signal both returns and is
  // heard. Drawn as a continuous band rather than one bar per sample, so a
  // reader sees a region rather than the sampling grid.
  let band = [];
  const flush = () => {
    if (band.length > 1) {
      const top = band.map(s => `${xOf(s.distance_km).toFixed(1)},${yOf(s.muf_mhz).toFixed(1)}`);
      const bottom = band.slice().reverse()
        .map(s => `${xOf(s.distance_km).toFixed(1)},${yOf(s.lof_mhz).toFixed(1)}`);
      svg.append(el('polygon', { points: top.concat(bottom).join(' '),
                                 fill: 'rgba(84,209,139,.14)' }));
    }
    band = [];
  };
  for (const s of S) { if (s.usable) band.push(s); else flush(); }
  flush();
  for (const r of runsOf(S, s => s.muf_mhz)) polyline(svg, r.map(([x, y]) => [xOf(x), yOf(y)]), '#45c7e0', 1.6);
  for (const r of runsOf(S, s => s.lof_mhz)) polyline(svg, r.map(([x, y]) => [xOf(x), yOf(y)]), '#efa14e', 1.6);

  polyline(svg, [[pad.l, yOf(frequency)], [W - pad.r, yOf(frequency)]], '#c6d5e3', 1, '2 3');
  svg.append(el('text', { x: W - pad.r, y: yOf(frequency) - 3, fill: '#c6d5e3',
                          'font-size': 9, 'text-anchor': 'end' }, 'working ' + frequency.toFixed(2) + ' MHz'));
  if (target >= xmin && target <= xmax) {
    polyline(svg, [[xOf(target), pad.t], [xOf(target), H - pad.b]], '#8aa0b5', 1, '3 3');
  }
}

/* ------------------------------------------------------------ sidebar ui */

function buildSidebar() {
  const host = document.getElementById('sidebar');
  host.textContent = '';
  for (const group of GROUPS) {
    const sec = document.createElement('section');
    sec.className = 'group'; sec.dataset.open = group.open ? '1' : '0';
    const h = document.createElement('h2');
    h.innerHTML = `<span>${group.title}</span><span class="caret">▾</span>`;
    h.onclick = () => sec.dataset.open = sec.dataset.open === '1' ? '0' : '1';
    sec.append(h);
    const body = document.createElement('div'); body.className = 'body';
    for (const item of group.items) body.append(buildControl(item));
    sec.append(body); host.append(sec);
  }
}

function buildControl(item) {
  const wrap = document.createElement('div'); wrap.className = 'ctl';

  if (item.bands) {
    const top = document.createElement('div'); top.className = 'top';
    top.innerHTML = '<label>Band shortcuts</label>';
    const chips = document.createElement('div'); chips.className = 'chips';
    for (const [name, mhz] of BANDS) {
      const b = document.createElement('button');
      b.className = 'tiny'; b.textContent = name; b.dataset.mhz = mhz;
      b.onclick = () => { state.frequency_mhz = mhz; syncInputs(); refresh(); };
      chips.append(b);
    }
    wrap.append(top, chips); return wrap;
  }

  const top = document.createElement('div'); top.className = 'top';
  const label = document.createElement('label'); label.textContent = item.label;
  const val = document.createElement('span'); val.className = 'val';
  val.dataset.for = item.k;
  top.append(label, val); wrap.append(top);

  if (item.seg) {
    const seg = document.createElement('div'); seg.className = 'seg';
    for (const [value, text] of item.seg) {
      const b = document.createElement('button');
      b.textContent = text; b.dataset.k = item.k; b.dataset.v = value;
      b.onclick = () => { state[item.k] = value; syncInputs(); refresh(); };
      seg.append(b);
    }
    val.remove(); wrap.append(seg); return wrap;
  }
  if (item.select) {
    const sel = document.createElement('select'); sel.dataset.k = item.k;
    for (const [value, text] of item.select) {
      const o = document.createElement('option'); o.value = value; o.textContent = text;
      sel.append(o);
    }
    sel.onchange = () => { state[item.k] = sel.value; refresh(); };
    val.remove(); wrap.append(sel); return wrap;
  }
  if (item.datetime) {
    const inp = document.createElement('input');
    inp.type = 'datetime-local'; inp.dataset.k = item.k;
    inp.oninput = () => { if (inp.value) { state[item.k] = inp.value; refresh(); } };
    val.remove(); wrap.append(inp); return wrap;
  }

  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = item.min; slider.max = item.max;
  slider.step = item.step; slider.dataset.k = item.k;
  slider.oninput = () => {
    const v = parseFloat(slider.value);
    if (item.moves) moveReceiverTo(v); else state[item.k] = v;
    syncInputs(); refresh();
  };
  wrap.append(slider);
  return wrap;
}

/* Dragging the distance slides the receiver along the current bearing, so
   the two ways of saying where it is stay one thing rather than two. */
function moveReceiverTo(km) {
  const bearing = bearingDeg([state.tx_lat, state.tx_lon], [state.rx_lat, state.rx_lon]);
  const p = destination([state.tx_lat, state.tx_lon], bearing, km);
  state.rx_lat = p[0]; state.rx_lon = p[1]; state.distance_km = km;
}

function syncInputs() {
  state.distance_km = greatCircleKm([state.tx_lat, state.tx_lon], [state.rx_lat, state.rx_lon]);
  for (const group of GROUPS) for (const item of group.items) {
    if (item.bands) continue;
    const v = state[item.k];
    const span = document.querySelector(`.val[data-for="${item.k}"]`);
    if (span) span.innerHTML = (typeof v === 'number' ? v.toFixed(item.dec ?? 1) : v) +
      (item.unit ? `<span class="u">${item.unit}</span>` : '');
    const slider = document.querySelector(`input[type=range][data-k="${item.k}"]`);
    if (slider && typeof v === 'number') slider.value = v;
    const sel = document.querySelector(`select[data-k="${item.k}"]`);
    if (sel) sel.value = v;
    const dt = document.querySelector(`input[data-k="${item.k}"][type=datetime-local]`);
    if (dt) dt.value = v;
    document.querySelectorAll(`.seg button[data-k="${item.k}"]`)
      .forEach(b => b.classList.toggle('on', b.dataset.v === v));
  }
  document.querySelectorAll('.chips .tiny[data-mhz]').forEach(b =>
    b.classList.toggle('on', Math.abs(parseFloat(b.dataset.mhz) - state.frequency_mhz) < 0.02));
}

/* --------------------------------------------------------------- readout */

function row(name, value, unit, tone) {
  const cls = tone ? ` ${tone}` : '';
  return `<div class="row"><span class="dot${cls}"></span><span class="n">${name}</span>` +
         `<span class="v${cls}">${value}${unit ? `<span class="u">${unit}</span>` : ''}</span></div>`;
}
function renderReadout(d) {
  const host = document.getElementById('readout');
  if (!d) { host.innerHTML = '<div class="status"><div class="label">…</div></div>'; return; }
  const i = d.ionosphere, r = d.ray, b = d.budget, g = d.geometry, w = d.ground_wave;
  const marginTone = m => m === null || m === undefined ? 'bad' : m >= 10 ? 'good' : m >= 0 ? 'warn' : 'bad';

  let html = `<div class="status ${d.status.tone}">
    <div class="label">${d.status.label}</div><div class="note">${d.status.note}</div></div>`;

  html += `<div class="rgroup"><h4>Ionosphere</h4>` +
    row('foF2 (local)', fmt(i.fof2_local_mhz, 2), 'MHz', 'on') +
    row('foF2 (path)', fmt(i.fof2_path_mhz, 2), 'MHz') +
    row('hmF2', fmt(i.hmf2_km, 0), 'km') +
    row('foE', fmt(i.foe_mhz, 2), 'MHz') +
    row('foD', fmt(i.fod_mhz, 3), 'MHz') +
    row('TEC (to 600 km)', fmt(i.tec_tecu, 1), 'TECU') + `</div>`;

  html += `<div class="rgroup"><h4>Ray at ${fmt(state.launch_angle_deg, 1)}°</h4>` +
    row('Ray state', r.state, '', r.state === 'reflected' ? 'good' : 'bad') +
    row('Apex altitude', fmt(r.apex_height_km, 0), 'km') +
    row('Virtual height', fmt(r.virtual_height_km, 0), 'km') +
    row('Landing / hop', fmt(r.hop_range_km, 0), 'km') +
    row('Max range', fmt(r.max_range_km, 0), 'km') +
    row('Group delay', fmt(r.delay_ms, 2), 'ms') + `</div>`;

  html += `<div class="rgroup"><h4>Budget</h4>`;
  if (b) {
    html += row('Mode used', `${b.hops} hop${b.hops > 1 ? 's' : ''} · ${b.mode}`, '') +
      row('Take-off angle', fmt(b.launch_elevation_deg, 1), '°') +
      row('Ionospheric loss', fmt(b.ionospheric_loss_db, 1), 'dB', 'on') +
      row('Ground loss', fmt(b.ground_loss_db, 1), 'dB') +
      row('Field strength', fmt(b.field_strength_dbuv_m, 1), 'dBµV/m') +
      row('Received power', fmt(b.received_power_dbm, 1), 'dBm') +
      row('Noise floor', fmt(b.noise_floor_dbm, 1), 'dBm') +
      row('SNR', fmt(b.snr_db, 1), 'dB', marginTone(b.snr_db - state.required_snr_db)) +
      row('Fade margin (90%)', '−' + fmt(b.fade_margin_db, 1), 'dB') +
      row('Link margin', fmt(b.effective_margin_db, 1), 'dB', marginTone(b.effective_margin_db));
  } else {
    html += row('Skywave', 'no ray reaches the receiver', '', 'bad');
  }
  html += `</div>`;

  // The surface route, shown whether or not it wins. Inside the skip zone
  // it is the only thing there is; outside it, seeing how far down it sits
  // is the point.
  if (w) {
    html += `<div class="rgroup"><h4>Ground wave</h4>`;
    if (w.significant) {
      html += row('Over sea', fmt(w.sea_fraction * 100, 0), '%', 'on') +
        row('Surface loss', fmt(w.surface_loss_db, 1), 'dB') +
        row('Curvature loss', fmt(w.curvature_loss_db, 1), 'dB') +
        row('Received power', fmt(w.received_power_dbm, 1), 'dBm') +
        row('Group delay', fmt(w.delay_ms, 2), 'ms') +
        row('Link margin', fmt(w.margin_db, 1), 'dB', marginTone(w.margin_db));
    } else {
      // Past this point the numbers stop meaning anything: the loss may
      // have hit the model's own cap, and either way the surface route is
      // further under the requirement than any measure could recover.
      html += row('Surface route',
                  w.saturated ? 'out of range' : fmt(-w.margin_db, 0) + ' dB short',
                  '', 'bad');
    }
    html += `</div>`;
  }

  html += `<div class="rgroup"><h4>Path</h4>` +
    row('Distance', fmt(g.distance_km, 0), 'km') +
    row('Bearing', fmt(g.bearing_deg, 1), '°') +
    row('Path sunlit', fmt(g.sunlit_fraction * 100, 0), '%') +
    row('Terminator', g.crosses_terminator ? 'crossed' : 'no', '',
        g.crosses_terminator ? 'warn' : '') +
    row('Geomagnetic lat', fmt(g.geomagnetic_latitude_deg, 1), '°') +
    row('Gyrofrequency', fmt(g.gyrofrequency_mhz, 2), 'MHz') + `</div>`;
  host.innerHTML = html;
}

function renderFeed(d) {
  const sw = d.space_weather;
  document.getElementById('feedTitle').textContent =
    `Manual mode · ${sw.source} values`;
  document.getElementById('feedGrid').innerHTML =
    `<div><div class="k">F10.7 flux</div><div class="v">${fmt(sw.f107, 0)}<span class="u"> sfu</span></div></div>
     <div><div class="k">Kp now</div><div class="v">${fmt(sw.kp, 2)}</div></div>
     <div><div class="k">X-ray class</div><div class="v">${sw.xray_class}</div></div>
     <div><div class="k">Flare state</div><div class="v">${sw.is_flare ? 'flare' : 'quiet'}</div></div>
     <div><div class="k">Sunspots</div><div class="v">${fmt(sw.sunspot_number, 0)}</div></div>
     <div><div class="k">Storm</div><div class="v">${sw.is_storm ? 'storm' : 'quiet'}</div></div>`;
  document.getElementById('feedFoot').innerHTML =
    `TX ${fmt(d.geometry.tx[0], 2)}° ${fmt(d.geometry.tx[1], 2)}° · RX ${fmt(d.geometry.rx[0], 2)}° ${fmt(d.geometry.rx[1], 2)}°<br>` +
    `${fmt(d.geometry.distance_km, 0)} km · local solar time ${fmt(d.geometry.local_solar_hour, 1)} h<br>` +
    `Drag the globe to rotate · scroll to zoom`;

  // A sparkline of the ionosphere the path actually sees, sampled by hour.
  const svg = document.getElementById('spark'); svg.textContent = '';
  const pts = sparkHistory.map((v, i) => [i / Math.max(sparkHistory.length - 1, 1) * 220, v]);
  if (pts.length > 1) {
    const vs = sparkHistory, lo = Math.min(...vs), hi = Math.max(...vs);
    const span = Math.max(hi - lo, 0.5);
    const scaled = pts.map(([x, v]) => [x, 30 - (v - lo) / span * 26]);
    polyline(svg, scaled, '#45c7e0', 1.4);
    svg.append(el('text', { x: 1, y: 8, fill: '#5d7488', 'font-size': 8 }, hi.toFixed(1)));
    svg.append(el('text', { x: 1, y: 32, fill: '#5d7488', 'font-size': 8 }, lo.toFixed(1)));
    svg.append(el('text', { x: 219, y: 8, fill: '#5d7488', 'font-size': 8, 'text-anchor': 'end' },
                  'foF2 path, last ' + sparkHistory.length));
  }
}

/* ---------------------------------------------------------------- wiring */

const globe = new Globe(document.getElementById('globe'));
let linkData = null, coverageData = null, bandData = null;
let sparkHistory = [];
let fastToken = 0, slowTimer = null;

function payload() {
  const station = (gain, height) => ({
    lat: 0, lon: 0, antenna_type: state.antenna_type, height_m: height,
    ground: state.ground, extra_gain_dbi: gain,
    design_frequency_mhz: state.frequency_mhz,
    power_w: state.power_w, bandwidth_hz: state.bandwidth_hz,
    noise_figure_db: state.noise_figure_db, required_snr_db: state.required_snr_db,
  });
  const tx = Object.assign(station(state.tx_gain, state.tx_height_m),
                           { lat: state.tx_lat, lon: state.tx_lon, name: 'TX' });
  const rx = Object.assign(station(state.rx_gain, state.rx_height_m),
                           { lat: state.rx_lat, lon: state.rx_lon, name: 'RX' });
  return {
    transmitter: tx, receiver: rx, when: state.when + ':00Z',
    space_weather: { f107: state.f107, kp: state.kp, sunspot_number: state.sunspot_number },
    frequency_mhz: state.frequency_mhz, launch_angle_deg: state.launch_angle_deg,
    magnetoionic_mode: state.mode, max_hops: state.max_hops,
    fof2_scale: state.fof2_scale, hmf2_offset_km: state.hmf2_offset_km,
  };
}

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(x => `${(x.loc || []).slice(-1)}: ${x.msg}`).join('; ')
      : (data.detail || 'request rejected');
    throw new Error(detail);
  }
  return data;
}

function showError(message) {
  let box = document.querySelector('.err');
  if (!message) { if (box) box.remove(); return; }
  if (!box) { box = document.createElement('div'); box.className = 'err';
              document.querySelector('.stage').append(box); }
  box.textContent = message;
}

async function refreshFast() {
  const token = ++fastToken;
  const started = performance.now();
  try {
    const d = await post('/api/link', payload());
    if (token !== fastToken) return;          // a newer request already won
    linkData = d; showError(null);
    document.getElementById('computeChip').textContent =
      (performance.now() - started).toFixed(0) + ' ms';
    document.getElementById('simClock').textContent =
      d.when.replace('T', ' ').slice(0, 16) + ' UTC';
    sparkHistory.push(d.ionosphere.fof2_path_mhz);
    if (sparkHistory.length > 60) sparkHistory.shift();
    renderReadout(d); renderFeed(d); globe.draw(d);
    drawBand(document.getElementById('chartBand'), bandData,
             d.geometry.distance_km, state.frequency_mhz);
  } catch (e) {
    if (token === fastToken) showError(String(e.message || e));
  }
}

function refreshSlow() {
  clearTimeout(slowTimer);
  slowTimer = setTimeout(async () => {
    const body = payload();
    const busyC = document.getElementById('busyCoverage');
    const busyB = document.getElementById('busyBand');
    busyC.hidden = false; busyB.hidden = false;
    const target = () => (linkData ? linkData.geometry.distance_km : state.distance_km);
    post('/api/coverage', body).then(d => {
      coverageData = d; busyC.hidden = true;
      drawCoverage(document.getElementById('chartCoverage'), d, target());
    }).catch(() => { busyC.hidden = true; });
    post('/api/band', body).then(d => {
      bandData = d; busyB.hidden = true;
      drawBand(document.getElementById('chartBand'), d, target(), state.frequency_mhz);
    }).catch(() => { busyB.hidden = true; });
  }, 450);
}

function refresh() { refreshFast(); refreshSlow(); }

function buildPresets() {
  const host = document.getElementById('presets');
  for (const name in PRESETS) {
    const b = document.createElement('button');
    b.className = 'preset'; b.textContent = name;
    b.onclick = () => {
      Object.assign(state, PRESETS[name]);
      host.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      syncInputs(); refresh();
      if (linkData) globe.framePath(linkData.geometry);
    };
    host.append(b);
  }
  const reset = document.createElement('button');
  reset.className = 'preset'; reset.textContent = 'Reset';
  reset.onclick = () => location.reload();
  host.append(reset);
  host.querySelectorAll('button')[1].classList.add('on');
}

/* The LIVE button pulls current drivers from NOAA SWPC through the backend,
   which already handles timeouts, retries, alternate endpoints and a cache.
   Egress is blocked in some environments, so a failure is reported as such
   rather than silently leaving stale manual numbers on screen looking live. */
async function toggleLive() {
  const btn = document.getElementById('liveBtn');
  if (state.live) {
    state.live = false;
    btn.classList.remove('on'); btn.textContent = '◎ LIVE off';
    document.getElementById('modeText').textContent = 'manual';
  document.getElementById('liveBtn').onclick = toggleLive;
    return;
  }
  btn.textContent = '◎ fetching…'; btn.disabled = true;
  try {
    const res = await fetch('/api/space-weather');
    const sw = await res.json();
    if (!res.ok) throw new Error('feed unavailable');
    state.f107 = sw.f107; state.kp = sw.kp; state.sunspot_number = sw.sunspot_number;
    state.live = true;
    btn.classList.add('on'); btn.textContent = '◎ LIVE on';
    document.getElementById('modeText').textContent = sw.source;
    syncInputs(); refresh();
  } catch (e) {
    btn.textContent = '◎ LIVE n/a';
    showError('Live feed unreachable — staying on manual values. ' +
              'Outbound access to services.swpc.noaa.gov is required.');
    setTimeout(() => { btn.textContent = '◎ LIVE off'; showError(null); }, 6000);
  } finally {
    btn.disabled = false;
  }
}

function boot() {
  buildSidebar(); buildPresets(); syncInputs();
  document.querySelectorAll('.layers input[data-layer]').forEach(box => {
    box.onchange = () => { layers[box.dataset.layer] = box.checked; globe.draw(); };
  });
  document.getElementById('framePath').onclick = () => {
    if (linkData) globe.framePath(linkData.geometry);
  };
  document.getElementById('modeText').textContent = 'manual';
  document.getElementById('liveBtn').onclick = toggleLive;
  fetch('/api/land').then(r => r.json()).then(d => { globe.setLand(d.polygons); globe.draw(); });
  const redraw = () => {
    globe.draw();
    drawCoverage(document.getElementById('chartCoverage'), coverageData,
                 linkData ? linkData.geometry.distance_km : state.distance_km);
    drawBand(document.getElementById('chartBand'), bandData,
             linkData ? linkData.geometry.distance_km : state.distance_km, state.frequency_mhz);
  };
  window.addEventListener('resize', redraw);
  refresh();
}
boot();

---
name: frontend-development
description: Guidelines and constraints for modifying the React, Vite, Tailwind CSS 4.x, and Recharts frontend of BOagent. Use this skill whenever the user asks to modify App.tsx, BenchMode.tsx, OperationalMode.tsx, custom components in src/components/, or the styling theme in index.css.
---

# Frontend Development & Design Skill (frontend-development)

Use this skill when editing frontend elements, implementing new interactive pages, adjusting visual styles, or configuring Recharts visualizations.

---

## 1. Trigger Scopes & Scenarios
- Modifying modes, root layouts, or routing switchers in `/home/dministrator/project/BOagent/frontend/src/App.tsx`.
- Modifying benchmark runner views, chart integrations, or state aggregation in `/home/dministrator/project/BOagent/frontend/src/BenchMode.tsx`.
- Modifying experimental form fields, canvas renders, or history logs in `/home/dministrator/project/BOagent/frontend/src/OperationalMode.tsx`.
- Modifying reusable UI blocks under `/home/dministrator/project/BOagent/frontend/src/components/` (e.g., `AcquisitionConfig.tsx`, `ConvergenceChart.tsx`, `LandscapeCanvas.tsx`, `MetricReadout.tsx`).
- Modifying Axios api clients or SSE stream processing in `/home/dministrator/project/BOagent/frontend/src/lib/api.ts`.
- Adjusting styling values, theme variables, or animations in `/home/dministrator/project/BOagent/frontend/src/index.css`.

---

## 2. Key Files to Read First
- `/home/dministrator/project/BOagent/frontend/src/App.tsx`: Mode switcher and main layout.
- `/home/dministrator/project/BOagent/frontend/src/BenchMode.tsx`: State manager for streaming benchmark data.
- `/home/dministrator/project/BOagent/frontend/src/OperationalMode.tsx`: human-in-the-loop experiment interface.
- `/home/dministrator/project/BOagent/frontend/src/lib/api.ts`: Stream parser and client configurations.
- `/home/dministrator/project/BOagent/frontend/src/index.css`: Tailwind CSS 4 theme rules and custom animations.

---

## 3. Recommended Workflow & Architectural Rules

### 3.1 Styling & Theme (Tailwind CSS 4.x)
- **Tailwind 4 Engine**: CSS-first configuration. **Do NOT create or modify `tailwind.config.js` or `postcss.config.js`.** All extensions and overrides must reside in `src/index.css` inside the `@theme` block:
  ```css
  @theme {
    --color-graphite-900: #0f172a;
    --color-signal-500: #10b981;    /* Emerald for LLM BO indicator */
    --color-amber-500: #f59e0b;     /* Amber for Traditional BO indicator */
    --color-glass-bg: rgba(15, 23, 42, 0.45);
  }
  ```
- **Typography & Fonts**: Use Space Grotesk (for headers/display), Plus Jakarta Sans (for body), and JetBrains Mono (for code and logs).
- **Consistent Colors**:
  - Traditional BO: Amber/Orange (`text-amber-500`, `--color-amber-*`).
  - LLMBO/AI: Green/Emerald (`text-signal-500`, `--color-signal-*`).
  - Fault/Error: Red/Rose (`text-fault-400`, `--color-fault-*`).
- **Glassmorphism**: Always apply `.panel` and `.sub-panel` styles (utilizing `backdrop-filter: blur(16px)` and semi-transparent backgrounds) for container layouts to preserve the "Scientific Instrument" aesthetic.
- **Custom Animations**: `pulse-ring` (live status indicator), `value-flash` (data update highlight). Defined in `index.css` — reuse instead of creating new keyframes.
- **Hex Colors**: Avoid hardcoding hex colors in inline style blocks or Recharts elements. You must explicitly advise against using raw hex colors and instead reference Tailwind v4 custom properties (e.g., `var(--color-signal-500)`) for UI styling and Recharts Line coloring.
- **Vite Integration**: Uses `@tailwindcss/vite` plugin. Do NOT add `postcss.config.js` or `tailwind.config.js`.

### 3.2 Charting & Visualization (Recharts)
- **Animation Disabling**: All `<Line>`, `<Area>`, and `<Scatter>` components inside `ConvergenceChart.tsx` and `LandscapeCanvas.tsx` must set `isAnimationActive={false}`. This avoids re-drawing/bouncing charts from zero when receiving frequent SSE updates.
- **Legends & Interaction**: Custom legends in `ConvergenceChart` should allow toggling the visibility of Traditional vs. LLMBO series.
- **Dynamic Domains**: Y-axes must calculate domains dynamically with padding (rather than static bounds) to adapt cleanly to different dataset scales.
- **Responsiveness**: Always wrap charts in `<ResponsiveContainer>`. Set explicit heights on parent container elements (e.g., `h-[420px]`) because `<ResponsiveContainer>` cannot resolve raw dynamic height from a flexible parent.

### 3.3 State Management & API Communications
- **State Separation**: Keep states for `BenchMode` and `OperationalMode` completely separated. Both components are kept rendered with `display: none` toggles; combining states will lead to rendering synchronization conflicts.
- **SSE Stream Decoding**: The custom `streamComparison` in `src/lib/api.ts` decodes chunked JSON streams. It splits raw bytes by `\n\n`. SSE event types: `meta`, `seed_start`, `step_start`, `aggregate`, `done`. Ensure your messages match this framing.
- **Input Pre-validation**: Validate range constraints (`min <= max` and positive trap densities) on the client side inside `OperationalMode.tsx` before sending suggest requests.
- **Declarative Color Maps**: Use lookup objects (e.g., `typeColorMap` in `LandscapeCanvas.tsx`) instead of nested ternaries for conditional styling.

---

## 4. Key Verification Commands
```bash
cd /home/dministrator/project/BOagent/frontend
npm run build              # Verifies typescript compiler (tsc -b) & vite build
```

---

## 5. Prohibited Actions (Red Lines)
> [!WARNING]
> Do not use utility classes from Tailwind v3 that have been deprecated or modified in v4 without verifying their existence in `index.css`.
> Do not add external heavy state managers (Redux, MobX, Zustand) without explicit user instructions.

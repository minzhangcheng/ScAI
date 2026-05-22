# ScAI Frontend

React + Vite + TypeScript + Zustand frontend for the ScAI single-cell analysis platform.

## Security guidelines

### Never render LLM output or backend HTML via `dangerouslySetInnerHTML`

Any string that originates from an LLM agent, the analysis sandbox, or backend-generated
HTML reports is **untrusted** and must NEVER be passed to React's
`dangerouslySetInnerHTML`. This is enforced by convention — there are currently no
`dangerouslySetInnerHTML` uses in this codebase, and that should remain true.

If you need to display a backend-generated HTML report (e.g., the exploration report
or detailed report), embed it in a **sandboxed iframe**:

```tsx
<iframe
  src={reportUrl}
  sandbox="allow-scripts allow-same-origin"  // adjust scopes minimally per use case
  style={{ width: '100%', height: '80vh', border: 0 }}
/>
```

Or render markdown-like content using a sanitizing renderer (e.g., `react-markdown`
with allow-listed components — never `rehype-raw` on untrusted input).

### Token caching

JWT tokens are cached in `frontend/src/utils/authStorage.ts` via a module-level
`_cachedToken`. The cache is invalidated by `setToken` and `clearToken`. All logout
paths (in `useApi` and the app store) must route through `clearToken` so the
in-memory copy is also cleared — never call `localStorage.removeItem` directly.

## Tech stack

- React 18 + TypeScript
- Vite for dev/build
- Zustand for state
- Native WebSocket for backend streaming

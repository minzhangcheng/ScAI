import { describe, expect, it } from 'vitest'

import appSource from './App.tsx?raw'
import layoutSource from './components/app/AppLayout.tsx?raw'

describe('App layout decomposition contract', () => {
  it('keeps mounted-tab ownership and resize effects in App', () => {
    expect(appSource).toContain('mountedTabsRef.current.add(activeTab)')
    expect(appSource).toContain('Math.max(20, Math.min(80, newWidth))')
    expect(appSource).toContain("document.addEventListener('mousemove', handleMouseMove)")
  })

  it('keeps lazy panel mounting and the split-pane DOM in AppLayout', () => {
    for (const panel of ['AnalysisPanel', 'DataPanel', 'PromptsPanel', 'ResultsPanel', 'ReportPanel', 'ChatPanel', 'MessageStream']) {
      expect(layoutSource).toContain(`const ${panel} = lazy(`)
    }
    expect(layoutSource).toContain("mountedTabsRef.current.has('analysis')")
    expect(layoutSource).toContain("style={{ width: `${messagePanelWidth}%` }}")
    expect(layoutSource).toContain('onMouseDown={() => setIsResizing(true)}')
  })
})

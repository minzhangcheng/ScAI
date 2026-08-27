import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import appSource from '../App.tsx?raw'
import mainSource from '../main.tsx?raw'
import analysisControlsSource from '../components/analysisPanel/AnalysisControls.tsx?raw'
import analysisDialogsSource from '../components/analysisPanel/AnalysisDialogs.tsx?raw'
import analysisPanelImplSource from '../components/analysisPanel/AnalysisPanelImpl.tsx?raw'
import analysisStatusSource from '../components/analysisPanel/AnalysisStatusView.tsx?raw'
import authLayoutSource from '../components/AuthLayout.tsx?raw'
import chatPanelSource from '../components/ChatPanel.tsx?raw'
import figureGallerySource from '../components/analysis/FigureGallery.tsx?raw'
import graphicAbstractSource from '../components/analysis/GraphicAbstract.tsx?raw'
import loadingScreenSource from '../components/LoadingScreen.tsx?raw'
import messageItemSource from '../components/messageStream/MessageItem.tsx?raw'
import messageRenderersSource from '../components/messageStream/renderers.tsx?raw'
import messageStreamViewSource from '../components/messageStream/MessageStreamView.tsx?raw'
import resultsPanelSource from '../components/ResultsPanel.tsx?raw'
import sidebarResizeHandleSource from '../components/SidebarResizeHandle.tsx?raw'
import settingsDialogSource from '../components/SettingsDialog.tsx?raw'
import indexSource from '../../index.html?raw'
import tailwindConfigSource from '../../tailwind.config.js?raw'

const globalsSource = readFileSync(join(process.cwd(), 'src/styles/globals.css'), 'utf8')
const analysisPanelSource = [
  analysisControlsSource,
  analysisDialogsSource,
  analysisPanelImplSource,
  analysisStatusSource,
].join('\n')
const themeOwnedPaths = [
  'src/components/ErrorBoundary.tsx',
  'src/components/LanguageSelector.tsx',
  'src/components/PromptsPanel.tsx',
  'src/components/UserMenu.tsx',
  'src/components/analysis/DirectionEvidence.tsx',
  'src/components/analysis/DownloadButtons.tsx',
  'src/components/analysis/IntegratedReport.tsx',
  'src/components/analysis/RoundSummaries.tsx',
  'src/components/data/DataPanel.tsx',
  'src/components/data/DataSection.tsx',
  'src/components/data/GroupManager.tsx',
  'src/components/data/IndicatorForm.tsx',
  'src/components/shared/CustomTypeInput.tsx',
  'src/components/shared/ErrorBanner.tsx',
  'src/components/sidebar/NewSessionForm.tsx',
  'src/components/snapshots/RuntimeWarningBanner.tsx',
  'src/components/snapshots/SnapshotPreview.tsx',
  'src/components/snapshots/SnapshotTree.tsx',
  'src/components/snapshots/StorageSummary.tsx',
  'src/constants/agents.ts',
  'src/pages/Admin.tsx',
  'src/pages/ApiKeys.tsx',
  'src/pages/Login.tsx',
  'src/pages/Register.tsx',
  'src/pages/Setup.tsx',
]

describe('system color theme contract', () => {
  it('uses browser media authority and no persisted override', () => {
    expect(tailwindConfigSource).toContain("darkMode: 'media'")
    expect(indexSource).toContain('<meta name="color-scheme" content="light dark" />')
    expect(globalsSource).toContain('@media (prefers-color-scheme: dark)')
    expect(globalsSource).toMatch(
      /@media \(prefers-color-scheme: dark\)\s*{\s*:root\s*{[^}]*color-scheme:\s*dark;/,
    )
    expect(appSource).not.toMatch(/localStorage\.(getItem|setItem)\(['"]theme/)
    expect(appSource).not.toMatch(/setTheme|toggleTheme/)
  })

  it('uses static classes and preserves light publication canvases', () => {
    expect(appSource).toContain('bg-surface-50 dark:bg-surface-950')
    expect(appSource).toContain('bg-white/90 dark:bg-surface-900/90')
    expect(appSource).toContain('bg-white dark:bg-surface-900')
    expect(mainSource).toContain('<SystemThemeProvider>')
    expect(messageRenderersSource).toContain('isDark ? oneDark : oneLight')
    expect(messageItemSource).toContain('bg-white dark:bg-surface-900')
    expect(messageStreamViewSource).toContain('bg-surface-50 dark:bg-surface-950')
    expect(messageStreamViewSource).toContain('bg-white/90 dark:bg-surface-900/90')
    expect(sidebarResizeHandleSource).toContain('bg-surface-50 dark:bg-surface-950')
    expect(sidebarResizeHandleSource).toContain('bg-surface-200 dark:bg-surface-700')
    expect(globalsSource).toMatch(/\.publication-canvas\s*{[\s\S]*background:\s*#fff;[\s\S]*color-scheme:\s*light;/)
  })

  it('keeps core panels, dialogs, and scientific figures dark-mode safe', () => {
    for (const source of [
      analysisPanelSource,
      resultsPanelSource,
      chatPanelSource,
      authLayoutSource,
      loadingScreenSource,
      settingsDialogSource,
    ]) {
      expect(source).toMatch(/dark:/)
    }
    expect(analysisPanelSource).toContain('dark:bg-surface-900')
    expect(resultsPanelSource).toContain('dark:bg-surface-900')
    expect(chatPanelSource).toContain('publication-canvas')
    expect(chatPanelSource).toContain('className="publication-canvas w-full h-auto"')
    expect(figureGallerySource).toContain('className="publication-canvas relative aspect-square cursor-pointer group"')
    expect(figureGallerySource).toContain('className="publication-canvas p-4 max-h-[60vh] overflow-auto"')
    expect(graphicAbstractSource).toContain('className="publication-canvas relative cursor-pointer group"')
    expect(graphicAbstractSource).toContain('className="publication-canvas max-w-full max-h-[85vh]')

    for (const path of themeOwnedPaths) {
      expect(readFileSync(join(process.cwd(), path), 'utf8'), path).toContain('dark:')
    }
  })
})

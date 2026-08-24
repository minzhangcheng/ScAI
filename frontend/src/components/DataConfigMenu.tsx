import { useState, useRef } from 'react'
import { apiFetch } from '../hooks/useApi'
import { Download, Upload, FolderOpen, ChevronDown, Loader2 } from 'lucide-react'

interface DataConfigMenuProps {
  sessionId: string | undefined
  onImported?: (sessionId: string) => void
  onExportSession?: () => void
  exporting?: boolean
  onImportSession?: () => void
  importing?: boolean
  readOnly?: boolean
}

export function DataConfigMenu({ sessionId, onImported, onExportSession, exporting, onImportSession, importing, readOnly }: DataConfigMenuProps) {
  const [open, setOpen] = useState(false)
  const [showExportWarning, setShowExportWarning] = useState(false)
  const [showImportConfirm, setShowImportConfirm] = useState(false)
  const [importFile, setImportFile] = useState<File | null>(null)
  const [importError, setImportError] = useState('')
  const [importingConfig, setImportingConfig] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleExport = async () => {
    if (!sessionId) return
    setShowExportWarning(false)
    try {
      const cfg = await apiFetch<Record<string, unknown>>(`/v1/analyses/${sessionId}/config/export`)
      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' })
      const a = document.createElement('a')
      const name = (cfg.analysis as Record<string, string>)?.name || 'scai-config'
      a.href = URL.createObjectURL(blob)
      a.download = `scai-config-${name}-${new Date().toISOString().slice(0, 16).replace(/[:-]/g, '')}.json`
      a.click()
    } catch (e) {
      console.error('Export failed:', e)
      alert('Export failed')
    }
  }

  const handleImport = async () => {
    if (!importFile) return
    setImportingConfig(true)
    setImportError('')
    try {
      const cfg = JSON.parse(await importFile.text())
      const result = await apiFetch<{ id: string }>('/v1/analyses/from-config', {
        method: 'POST',
        body: JSON.stringify(cfg),
      })
      setShowImportConfirm(false)
      setImportFile(null)
      if (onImported) onImported(result.id)
    } catch (e) {
      let msg = e instanceof Error ? e.message : 'Import failed'
      setImportError(msg)
    } finally {
      setImportingConfig(false)
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-surface-800 dark:text-surface-100 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg"
      >
        <FolderOpen className="w-4 h-4" />
        <span>Data/Config</span>
        <ChevronDown className="w-3 h-3 ml-auto" />
      </button>

      {open && (
        <div className="absolute bottom-full left-0 mb-1 w-52 bg-white dark:bg-surface-900 rounded-lg shadow-lg border border-surface-200 dark:border-surface-700 py-1 z-50">
          {/* Import Session */}
          {!readOnly && (
            <button
              onClick={() => { setOpen(false); onImportSession?.() }}
              disabled={importing}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-50 dark:hover:bg-surface-800 disabled:opacity-50"
            >
              {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              Import session
            </button>
          )}
          {/* Export Session */}
          <button
            onClick={() => { setOpen(false); onExportSession?.() }}
            disabled={!sessionId || exporting}
            className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-50 dark:hover:bg-surface-800 disabled:opacity-50"
          >
            {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Export session
          </button>
          {!readOnly && <div className="border-t border-surface-100 dark:border-surface-700 my-1" />}
          {/* Import Configure */}
          {!readOnly && (
            <button
              onClick={() => { setOpen(false); fileInputRef.current?.click() }}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-50 dark:hover:bg-surface-800"
            >
              <Upload className="w-4 h-4" /> Import configure
            </button>
          )}
          {/* Export Configure */}
          <button
            onClick={() => { setOpen(false); setShowExportWarning(true) }}
            disabled={!sessionId}
            className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-50 dark:hover:bg-surface-800 disabled:opacity-40"
          >
            <Download className="w-4 h-4" /> Export configure
          </button>
        </div>
      )}

      {/* Hidden file input for config import */}
      <input
        ref={fileInputRef}
        type="file"
        accept="application/json"
        className="hidden"
        onChange={e => {
          const f = e.target.files?.[0]
          if (f) { setImportFile(f); setShowImportConfirm(true) }
          e.target.value = ''
        }}
      />

      {/* Export Warning Dialog */}
      {showExportWarning && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-surface-900 rounded-xl p-6 max-w-md shadow-xl">
            <h3 className="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-3">Export Configuration</h3>
            <p className="text-red-600 text-sm mb-4">
              ⚠️ This file will contain all API keys. Do not share or commit it to version control.
            </p>
            <div className="flex justify-end gap-2">
              <button onClick={() => setShowExportWarning(false)} className="px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg">Cancel</button>
              <button onClick={handleExport} className="px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700">Export</button>
            </div>
          </div>
        </div>
      )}

      {/* Import Confirm Dialog */}
      {showImportConfirm && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-surface-900 rounded-xl p-6 max-w-md shadow-xl">
            <h3 className="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-3">Import Configuration</h3>
            <p className="text-sm text-surface-600 dark:text-surface-300 mb-2">This will create a new session with the imported configuration.</p>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">Data files are not imported — you must upload them after creation.</p>
            {importFile && <p className="text-sm font-mono bg-surface-50 dark:bg-surface-800 p-2 rounded mb-4">{importFile.name}</p>}
            {importError && <p className="text-red-600 text-sm mb-4">{importError}</p>}
            <div className="flex justify-end gap-2">
              <button onClick={() => { setShowImportConfirm(false); setImportFile(null); setImportError('') }} className="px-4 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg">Cancel</button>
              <button onClick={handleImport} disabled={importingConfig} className="px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50">{importingConfig ? 'Importing...' : 'Import'}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

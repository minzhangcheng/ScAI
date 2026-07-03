import { useState } from 'react'

export interface IntegrationCandidate {
  direction: string
  direction_id?: string
  [k: string]: unknown
}

export interface IntegrationGatePanelProps {
  cycle: number
  candidates: IntegrationCandidate[]
  onSubmit: (selected: IntegrationCandidate[]) => void
  onFinalize: () => void
}

export function IntegrationGatePanel({ cycle, candidates, onSubmit, onFinalize }: IntegrationGatePanelProps) {
  const [sel, setSel] = useState<Record<string, boolean>>({})
  const toggle = (id: string) => setSel((s) => ({ ...s, [id]: !s[id] }))
  const submit = () => onSubmit(candidates.filter((c, i) => sel[c.direction_id ?? String(i)]))
  return (
    <div className="integration-gate-panel">
      <h3>{`整合分析 (第 ${cycle} 次) — 选择新的研究方向`}</h3>
      <ul>
        {candidates.map((c, i) => {
          const id = c.direction_id ?? String(i)
          return (
            <li key={id}>
              <label>
                <input type="checkbox" checked={!!sel[id]} onChange={() => toggle(id)} />
                {c.direction}
              </label>
            </li>
          )
        })}
      </ul>
      <div className="actions">
        <button onClick={submit}>继续分析所选方向</button>
        <button onClick={onFinalize}>进入报告撰写 (Write report)</button>
      </div>
    </div>
  )
}

import { ChevronDown, ChevronUp } from 'lucide-react'
import type { DirectionLiteratureEvidence } from '../ExplorationReportDialog'

interface DirectionEvidenceProps {
  candidate: Record<string, any>
  evidence: DirectionLiteratureEvidence | undefined
  expanded: boolean
  onToggle: () => void
  language: 'zh' | 'en'
}

const qualLabels: Record<string, [string, string]> = {
  matched_species_tissue_available: ['匹配物种/组织证据可用', 'Matched species/tissue evidence available'],
  no_matched_species_tissue_evidence: ['无匹配物种/组织证据', 'No matched species/tissue evidence'],
  cross_tissue_only: ['仅跨组织证据', 'Cross-tissue evidence only'],
  cross_species_only: ['仅跨物种证据', 'Cross-species evidence only'],
}

export default function DirectionEvidence({
  candidate,
  evidence,
  expanded,
  onToggle,
  language,
}: DirectionEvidenceProps) {
  const isZh = language === 'zh'
  if (!evidence) return null
  const evidenceFailed = evidence.status === 'failed' || evidence.status === 'skipped'
  const speciesQual = evidence.species_tissue_qualification
  const qualPair = speciesQual ? qualLabels[speciesQual] : undefined
  const c = candidate

  const totalEvidence =
    (evidence.direct_evidence_count || 0) +
    (evidence.analogous_evidence_count || 0) +
    (evidence.contradictory_evidence_count || 0) +
    (evidence.methodological_precedent_count || 0)
  const timedOutWithoutEvidence = Boolean(evidence.timed_out) && totalEvidence === 0
  const partialWithEvidence = evidence.status === 'partial' && totalEvidence > 0

  return (
    <>
      <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
        {evidence.status === 'completed' && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">
            {isZh ? '证据已验证' : 'Evidence verified'}
          </span>
        )}
        {partialWithEvidence && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
            {isZh ? '部分证据' : 'Partial evidence'}
          </span>
        )}
        {timedOutWithoutEvidence && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">
            {isZh ? '检索超时' : 'Search timed out'}
          </span>
        )}
        {(evidence.status === 'failed' || evidence.status === 'skipped') && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
            {isZh ? '文献验证失败' : 'Literature verification failed'}
          </span>
        )}
        {evidence.search_tier === 'deep' && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
            {isZh ? '深度搜索' : 'Deep search used'}
          </span>
        )}
        {qualPair && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-700">
            {qualPair[isZh ? 0 : 1]}
          </span>
        )}
        {(evidence.negative_validation_confidence === 'medium' || evidence.negative_validation_confidence === 'high') && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-cyan-100 text-cyan-700">
            {isZh ? '负向验证' : 'Negative validation'}
          </span>
        )}
      </div>
      {c.risk_flags && c.risk_flags.length > 0 && (
        <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
          {c.risk_flags.map((flag: string, fi: number) => (
            <span key={fi} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-700">
              {flag === 'requires_new_experiment'
                ? (isZh ? '需新实验' : 'Requires new experiment')
                : flag}
            </span>
          ))}
        </div>
      )}
      {c.prerequisite_dependencies && c.prerequisite_dependencies.length > 0 && (
        <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-600">
            {isZh ? '未满足的前提条件' : 'Unmet prerequisite'}
          </span>
        </div>
      )}
      <div className="px-4 pb-1 pl-12">
        <button
          type="button"
          onClick={e => { e.stopPropagation(); onToggle() }}
          className="text-xs text-accent-600 hover:text-accent-700 flex items-center gap-1"
        >
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {isZh ? '查看文献证据' : 'View evidence'}
        </button>
      </div>
      {expanded && (
        <div className="mx-4 mb-4 ml-12 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-2">
          {timedOutWithoutEvidence && (
            <div className="text-slate-700 bg-slate-100 border border-slate-200 rounded p-2">
              {isZh ? '文献检索在保留引用前超时；该方向已保守评分。' : 'Literature search timed out before references could be retained; this direction was scored conservatively.'}
            </div>
          )}
          {partialWithEvidence && (
            <div className="text-amber-700 bg-amber-50 border border-amber-100 rounded p-2">
              {isZh ? '文献检索仅部分完成；下方引用是不完整结果。' : 'Literature search partially completed; references below are partial.'}
            </div>
          )}
          {evidenceFailed && (
            <div className="text-red-700 bg-red-50 border border-red-100 rounded p-2">
              {isZh ? '文献验证失败或已跳过；该方向已保守评分。' : 'Literature verification failed or was skipped; this direction was scored conservatively.'}
            </div>
          )}
          {evidence.summary && <div>{evidence.summary}</div>}
          <div className="grid grid-cols-2 gap-2 text-slate-500">
            {evidence.support_level && <div>{isZh ? '支持强度' : 'Support level'}: {evidence.support_level}</div>}
            {evidence.contradiction_level && <div>{isZh ? '反证强度' : 'Contradiction level'}: {evidence.contradiction_level}</div>}
            {evidence.prior_art_level && <div>{isZh ? '已有研究' : 'Prior art'}: {evidence.prior_art_level}</div>}
            {evidence.novelty_status && <div>{isZh ? '新颖性状态' : 'Novelty status'}: {evidence.novelty_status}</div>}
            {evidence.fallback_tier && <div>{isZh ? '降级层级' : 'Fallback tier'}: {evidence.fallback_tier}</div>}
          </div>
          {evidence.knowledge_gap_summary && (
            <div className="text-slate-700 bg-white border border-slate-200 rounded p-2">
              {evidence.knowledge_gap_summary}
            </div>
          )}
          {evidence.scheduler_limitation && (
            <div className="text-amber-700 bg-amber-50 border border-amber-100 rounded p-2">
              {evidence.scheduler_limitation}
            </div>
          )}
          <div className="grid grid-cols-2 gap-2">
            <div>{isZh ? '直接证据' : 'Direct'}: {evidence.direct_evidence_count || 0}</div>
            <div>{isZh ? '相似证据' : 'Analogous'}: {evidence.analogous_evidence_count || 0}</div>
            <div>{isZh ? '反证' : 'Contradictory'}: {evidence.contradictory_evidence_count || 0}</div>
            <div>{isZh ? '方法先例' : 'Methods'}: {evidence.methodological_precedent_count || 0}</div>
          </div>
          {[
            ...(evidence.direct_evidence || []),
            ...(evidence.analogous_evidence || []),
            ...(evidence.contradictory_evidence || []),
            ...(evidence.methodological_precedent || []),
          ].slice(0, 5).map(item => (
            <div key={item.pmid} className="border-t border-slate-200 pt-2">
              PMID: {item.pmid} — {item.title}
              {item.why_it_matters ? <span className="block text-slate-400">{item.why_it_matters}</span> : null}
            </div>
          ))}
          {(evidence.similar_published_directions || []).length > 0 && (
            <div className="border-t border-slate-200 pt-2">
              <div className="font-medium">{isZh ? '相似已发表方向' : 'Similar published directions'}</div>
              {(evidence.similar_published_directions || []).slice(0, 5).map((item, idx) => (
                <div key={idx} className="text-slate-500">
                  {item.pmid ? `PMID: ${item.pmid} — ` : ''}{item.title || item.url || 'Untitled reference'}
                </div>
              ))}
            </div>
          )}
          {evidence.confidence && (
            <div className="text-slate-500">{isZh ? '文献置信度' : 'Literature confidence'}: {evidence.confidence}</div>
          )}
          {(evidence.queries || []).length > 0 && (
            <div className="border-t border-slate-200 pt-2">
              <div className="font-medium">{isZh ? '检索式' : 'Queries'}</div>
              {(evidence.queries || []).slice(0, 5).map((q, qi) => (
                <div key={qi} className="text-slate-400 truncate">{q.category}: {q.query}</div>
              ))}
            </div>
          )}
          {(evidence.limitations || []).length > 0 && (
            <div className="border-t border-slate-200 pt-2">
              <div className="font-medium">{isZh ? '局限' : 'Limitations'}</div>
              {(evidence.limitations || []).map((lim, li) => (
                <div key={li} className="text-slate-400">{lim}</div>
              ))}
            </div>
          )}
          <div className="border-t border-slate-200 pt-2 grid grid-cols-2 gap-1 text-slate-500">
            <div>{isZh ? '生物学预期' : 'Biology expectedness'}: {c.biology_expectedness || 'N/A'}</div>
            <div>{isZh ? '数据新颖性' : 'Data novelty'}: {c.data_novelty || 'N/A'}</div>
            <div>{isZh ? '机制重要性' : 'Mechanistic importance'}: {c.mechanistic_importance || 'N/A'}</div>
            <div>{isZh ? '数据充分性' : 'Data sufficiency'}: {c.data_sufficiency || 'N/A'}</div>
            <div>{isZh ? '方法学缺口' : 'Methodological gap'}: {c.methodological_gap || 'N/A'}</div>
          </div>
        </div>
      )}
    </>
  )
}

import { useState } from 'react'
import { createPortal } from 'react-dom'
import { Info, X, Mail, Github } from 'lucide-react'
import { useAppStore } from '../stores/appStore'

export function AboutDialog() {
  const [open, setOpen] = useState(false)
  const { language } = useAppStore()
  const zh = language === 'zh'

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
        title={zh ? '关于' : 'About'}
      >
        <Info className="w-4 h-4" />
        <span>{zh ? '关于' : 'About'}</span>
      </button>

      {open && createPortal(
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}
        >
          <div className="bg-white rounded-2xl shadow-modal w-full max-w-md mx-4 max-h-[85vh] flex flex-col overflow-hidden animate-scale-in">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-accent-50 to-accent-100">
              <h2 className="text-lg font-display font-semibold text-slate-900">
                {zh ? '关于' : 'About'}
              </h2>
              <button
                onClick={() => setOpen(false)}
                className="p-1 hover:bg-slate-200 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>

            {/* Content */}
            <div className="px-6 py-4 space-y-4 overflow-y-auto">
              {/* Title */}
              <div className="text-center">
                <h3 className="text-2xl font-display font-bold text-accent-700">ScAI</h3>
                <p className="text-sm text-slate-500 mt-1">
                  {zh
                    ? '单细胞测序AI分析'
                    : 'Single Cell Sequencing Analysis Driven by AI'}
                </p>
                <div className="mt-2 inline-block px-3 py-1 bg-accent-100 text-accent-700 text-xs font-medium rounded-full">
                  v0.23.4
                </div>
              </div>

              <hr className="border-slate-200" />

              {/* Contact */}
              <div className="flex items-start gap-3">
                <Mail className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                <p className="text-sm text-slate-600">
                  {zh
                    ? '如有问题或建议，请联系程敏章（南昌大学第一附属医院，教育部类器官资源库），邮箱：'
                    : 'For questions or suggestions, please contact Minzhang Cheng (MOE Organoid Resource Center, The First Affiliated Hospital, Nanchang University). Email: '}
                  <a
                    href="mailto:mzcheng@ncu.edu.cn"
                    className="text-accent-600 hover:text-accent-700 hover:underline"
                  >
                    mzcheng@ncu.edu.cn
                  </a>
                </p>
              </div>

              {/* Source Code */}
              <div className="flex items-start gap-3">
                <Github className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                <div>
                  <div className="text-sm font-medium text-slate-700">
                    {zh ? '源代码' : 'Source Code'}
                  </div>
                  <a
                    href="https://github.com/minzhangcheng/scai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-accent-600 hover:text-accent-700 hover:underline"
                  >
                    github.com/minzhangcheng/scai
                  </a>
                </div>
              </div>

              <hr className="border-slate-200" />

              {/* Footer info */}
              <div className="text-center text-xs text-slate-400 space-y-1">
                <p>
                  {zh ? '更新日期：2026年5月7日' : 'Updated: May 7, 2026'}
                </p>
                <p>&copy; 2025-2026 Minzhang Cheng. All rights reserved.</p>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  )
}

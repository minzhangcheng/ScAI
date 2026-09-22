import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export function ChatMarkdown({ content }: { content: string }) {
  return (
    <div className="text-sm leading-relaxed break-words
      [&_p]:my-2 [&_h1]:text-xl [&_h2]:text-lg [&_h3]:text-base
      [&_h1]:font-bold [&_h2]:font-semibold [&_h3]:font-semibold
      [&_h1]:my-3 [&_h2]:my-3 [&_h3]:my-2
      [&_h4]:font-semibold [&_h5]:font-semibold [&_h6]:font-semibold
      [&_ul]:list-disc [&_ol]:list-decimal [&_ul]:pl-5 [&_ol]:pl-5
      [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-1
      [&_blockquote]:border-l-4 [&_blockquote]:border-slate-400 [&_blockquote]:pl-3 [&_blockquote]:my-3
      [&_a]:text-accent-600 dark:[&_a]:text-accent-400 [&_a]:underline
      [&_pre]:overflow-x-auto [&_pre]:rounded [&_pre]:bg-slate-200 dark:[&_pre]:bg-slate-900 [&_pre]:p-3 [&_pre]:my-3
      [&_code]:font-mono [&_code]:text-xs
      [&_th]:border [&_td]:border [&_th]:border-slate-300 [&_td]:border-slate-300
      dark:[&_th]:border-slate-600 dark:[&_td]:border-slate-600
      [&_th]:px-3 [&_td]:px-3 [&_th]:py-2 [&_td]:py-2 [&_th]:font-semibold
      [&_th]:bg-slate-200 dark:[&_th]:bg-slate-700
      [&_.katex-display]:block [&_.katex-display]:overflow-x-auto [&_.katex-display]:my-3
      [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { output: 'mathml', trust: false }]]}
        components={{
          table: ({ children }) => (
            <div className="my-3 max-w-full overflow-x-auto">
              <table className="min-w-full border-collapse text-left">{children}</table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

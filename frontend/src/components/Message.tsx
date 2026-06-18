import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { ChatMessage } from '../types'

type MessageProps = {
  message: ChatMessage
}

function citationLabel(source: string, index: number): string {
  const [documentName, chunkIndex] = source.split(':')
  if (!documentName) {
    return `${index + 1}. Source`
  }
  if (!chunkIndex) {
    return `${index + 1}. ${documentName}`
  }
  return `${index + 1}. ${documentName} - chunk ${chunkIndex}`
}

function copyCitation(source: string): void {
  void navigator.clipboard?.writeText(source)
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'
  const hasSources = Boolean(message.sources?.length)
  const confidence =
    typeof message.confidence === 'number' ? Math.round(message.confidence * 100) : null

  return (
    <article className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={[
          'max-w-[min(780px,92%)] rounded-2xl border px-4 py-3 shadow-sm',
          isUser
            ? 'border-cyan-400/30 bg-cyan-400/12 text-cyan-50'
            : 'border-neutral-800 bg-neutral-900/88 text-neutral-100',
          message.error ? 'border-red-400/40 bg-red-950/40' : '',
        ].join(' ')}
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <span className="text-xs font-semibold uppercase text-neutral-400">
            {isUser ? 'You' : 'Nexon Assistant'}
          </span>
          {message.pending && (
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-300" />
          )}
        </div>

        <div className="prose prose-invert max-w-none prose-p:my-2 prose-ul:my-2 prose-li:my-1 prose-a:text-cyan-300">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content || (message.pending ? 'Thinking...' : '')}
          </ReactMarkdown>
        </div>

        {!isUser && (hasSources || confidence !== null) && (
          <footer className="mt-4 border-t border-neutral-800 pt-3">
            {hasSources && (
              <div className="flex flex-wrap gap-2">
                {message.sources?.map((source, index) =>
                  source.startsWith('http') ? (
                    <a
                      className="rounded-full border border-neutral-700 bg-neutral-950 px-2.5 py-1 text-xs text-cyan-200 transition hover:border-cyan-400"
                      href={source}
                      key={source}
                      rel="noreferrer"
                      target="_blank"
                    >
                      {citationLabel(source, index)}
                    </a>
                  ) : (
                    <button
                      className="rounded-full border border-neutral-700 bg-neutral-950 px-2.5 py-1 text-left text-xs text-cyan-200 transition hover:border-cyan-400"
                      key={source}
                      onClick={() => copyCitation(source)}
                      title="Copy citation"
                      type="button"
                    >
                      {citationLabel(source, index)}
                    </button>
                  ),
                )}
              </div>
            )}

            {confidence !== null && (
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-neutral-400">
                <span className="rounded-full bg-neutral-950 px-2.5 py-1">
                  {confidence}% confidence
                </span>
                {typeof message.isGrounded === 'boolean' && (
                  <span
                    className={[
                      'rounded-full px-2.5 py-1',
                      message.isGrounded
                        ? 'bg-emerald-400/10 text-emerald-300'
                        : 'bg-amber-400/10 text-amber-300',
                    ].join(' ')}
                  >
                    {message.isGrounded ? 'Grounded' : 'Needs review'}
                  </span>
                )}
              </div>
            )}

            {Boolean(message.reasoningSteps?.length) && (
              <details className="mt-3 text-xs text-neutral-400">
                <summary className="cursor-pointer text-neutral-300">Reasoning</summary>
                <ol className="mt-2 list-decimal space-y-1 pl-4">
                  {message.reasoningSteps?.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </details>
            )}
          </footer>
        )}
      </div>
    </article>
  )
}

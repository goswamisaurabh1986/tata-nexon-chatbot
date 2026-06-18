import {
  type KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from 'react'

import { AdminPanel } from './AdminPanel'
import { Message } from './Message'
import { Sidebar } from './Sidebar'
import { useChat } from '../hooks/useChat'

const starterPrompts = [
  'What are the safety features of Tata Nexon?',
  'Tell me about Tata Nexon performance.',
  'Which Nexon features are useful for city driving?',
]

export function ChatWindow() {
  const chat = useChat()
  const [draft, setDraft] = useState('')
  const [activeView, setActiveView] = useState<'chat' | 'admin'>('chat')
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.messages, chat.isStreaming])

  const submit = (event?: { preventDefault: () => void }) => {
    event?.preventDefault()
    const nextMessage = draft.trim()
    if (!nextMessage || chat.isLoading) {
      return
    }
    chat.sendMessage(nextMessage)
    setDraft('')
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      submit(event)
    }
  }

  return (
    <div className="flex min-h-screen bg-neutral-950 text-neutral-100">
      <Sidebar
        activeThreadId={chat.activeThreadId}
        activeView={activeView}
        conversations={chat.conversations}
        onDeleteConversation={chat.deleteConversation}
        onNewConversation={() => {
          chat.newConversation()
          setActiveView('chat')
        }}
        onSelectView={setActiveView}
        onSelectConversation={(threadId) => {
          chat.selectConversation(threadId)
          setActiveView('chat')
        }}
      />

      {activeView === 'admin' ? (
        <AdminPanel isActive={activeView === 'admin'} />
      ) : (
      <main className="flex h-screen min-w-0 flex-1 flex-col bg-neutral-950">
        <header className="border-b border-neutral-800 bg-neutral-950/90 px-4 py-4 backdrop-blur sm:px-6">
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase text-cyan-300">
                Tata Nexon Chatbot
              </p>
              <h1 className="truncate text-xl font-semibold text-neutral-50">
                Ask the brochure
              </h1>
              <p className="truncate text-xs text-neutral-500">
                {chat.activeThreadId ?? 'New conversation'}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <label className="hidden items-center gap-2 rounded-full border border-neutral-800 px-3 py-1 text-xs text-neutral-300 sm:inline-flex">
                <input
                  checked={chat.streamingEnabled}
                  className="sr-only"
                  onChange={(event) => chat.setStreamingEnabled(event.target.checked)}
                  type="checkbox"
                />
                <span
                  className={[
                    'h-3.5 w-6 rounded-full p-0.5 transition',
                    chat.streamingEnabled ? 'bg-cyan-300' : 'bg-neutral-700',
                  ].join(' ')}
                >
                  <span
                    className={[
                      'block h-2.5 w-2.5 rounded-full bg-neutral-950 transition',
                      chat.streamingEnabled ? 'translate-x-2.5' : 'translate-x-0',
                    ].join(' ')}
                  />
                </span>
                Stream
              </label>
              <span className="hidden rounded-full border border-neutral-800 px-3 py-1 text-xs text-neutral-400 sm:inline-flex">
                {chat.isStreaming ? 'Streaming' : chat.isLoading ? 'Thinking' : 'Ready'}
              </span>
              <button
                className="rounded-xl border border-neutral-700 px-3 py-2 text-sm text-neutral-200 transition hover:border-cyan-400 hover:text-cyan-100 md:hidden"
                onClick={chat.newConversation}
                type="button"
              >
                New
              </button>
            </div>
          </div>
        </header>

        <section className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
          <div className="mx-auto flex max-w-5xl flex-col gap-5">
            {chat.messages.length === 0 ? (
              <div className="grid gap-3 pt-[12vh]">
                <div className="max-w-2xl">
                  <h2 className="text-3xl font-semibold text-neutral-50">
                    Tata Nexon answers with citations.
                  </h2>
                  <p className="mt-3 text-base leading-7 text-neutral-400">
                    Ask about safety, features, performance, variants, or brochure details.
                  </p>
                </div>
                <div className="mt-4 grid gap-2 sm:grid-cols-3">
                  {starterPrompts.map((prompt) => (
                    <button
                      className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-4 text-left text-sm text-neutral-200 transition hover:border-cyan-400/70 hover:bg-neutral-900"
                      key={prompt}
                      onClick={() => !chat.isLoading && chat.sendMessage(prompt)}
                      type="button"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              chat.messages.map((message) => <Message key={message.id} message={message} />)
            )}
            <div ref={bottomRef} />
          </div>
        </section>

        <footer className="border-t border-neutral-800 bg-neutral-950/95 px-4 py-4 sm:px-6">
          <form className="mx-auto max-w-5xl" onSubmit={submit}>
            {chat.error && (
              <div className="mb-3 rounded-xl border border-red-400/30 bg-red-950/40 px-4 py-2 text-sm text-red-100">
                {chat.error}
              </div>
            )}
            <div className="flex items-end gap-3 rounded-2xl border border-neutral-800 bg-neutral-900 p-2 shadow-2xl shadow-black/20 focus-within:border-cyan-400/70">
              <textarea
                className="max-h-40 min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 text-neutral-100 outline-none placeholder:text-neutral-500"
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about Tata Nexon..."
                rows={1}
                value={draft}
              />
              <button
                className="h-11 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-400"
                disabled={!draft.trim() || chat.isLoading}
                type="submit"
              >
                {chat.isLoading ? 'Wait' : 'Send'}
              </button>
            </div>
          </form>
        </footer>
      </main>
      )}
    </div>
  )
}

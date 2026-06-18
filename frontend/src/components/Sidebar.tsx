import type { Conversation } from '../types'

type SidebarProps = {
  conversations: Conversation[]
  activeThreadId: string | null
  activeView: 'chat' | 'admin'
  onDeleteConversation: (threadId: string) => void
  onNewConversation: () => void
  onSelectView: (view: 'chat' | 'admin') => void
  onSelectConversation: (threadId: string) => void
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

export function Sidebar({
  conversations,
  activeThreadId,
  activeView,
  onDeleteConversation,
  onNewConversation,
  onSelectView,
  onSelectConversation,
}: SidebarProps) {
  return (
    <aside className="hidden h-screen w-80 shrink-0 border-r border-neutral-800 bg-neutral-950/96 md:flex md:flex-col">
      <div className="border-b border-neutral-800 p-4">
        <div className="mb-4 grid grid-cols-2 rounded-xl border border-neutral-800 bg-neutral-900/70 p-1">
          {[
            ['chat', 'Chat'],
            ['admin', 'Admin'],
          ].map(([view, label]) => (
            <button
              className={[
                'rounded-lg px-3 py-2 text-sm font-medium transition',
                activeView === view
                  ? 'bg-cyan-300 text-neutral-950'
                  : 'text-neutral-400 hover:text-neutral-100',
              ].join(' ')}
              key={view}
              onClick={() => onSelectView(view as 'chat' | 'admin')}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>

        <button
          className="w-full rounded-xl border border-cyan-400/40 bg-cyan-400/12 px-4 py-3 text-sm font-semibold text-cyan-50 transition hover:border-cyan-300 hover:bg-cyan-400/18"
          onClick={onNewConversation}
          type="button"
        >
          New chat
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto p-3" aria-label="Chat history">
        {conversations.length === 0 ? (
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-4 text-sm text-neutral-400">
            No chats yet.
          </div>
        ) : (
          <div className="space-y-2">
            {conversations.map((conversation) => {
              const isActive = conversation.threadId === activeThreadId
              return (
                <div
                  className={[
                    'group rounded-xl border transition',
                    isActive
                      ? 'border-cyan-400/45 bg-cyan-400/10'
                      : 'border-transparent bg-neutral-900/50 hover:border-neutral-700',
                  ].join(' ')}
                  key={conversation.threadId}
                >
                  <button
                    className="block w-full px-3 py-3 text-left"
                    onClick={() => onSelectConversation(conversation.threadId)}
                    type="button"
                  >
                    <span className="line-clamp-2 text-sm font-medium text-neutral-100">
                      {conversation.title}
                    </span>
                    <span className="mt-1 block text-xs text-neutral-500">
                      {formatDate(conversation.updatedAt)}
                    </span>
                  </button>
                  <div className="flex justify-end px-3 pb-3">
                    <button
                      className="text-xs text-neutral-500 transition hover:text-red-300"
                      onClick={() => onDeleteConversation(conversation.threadId)}
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </nav>
    </aside>
  )
}

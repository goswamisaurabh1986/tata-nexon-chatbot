export type MessageRole = 'user' | 'assistant'

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  pending?: boolean
  error?: boolean
  sources?: string[]
  confidence?: number
  isGrounded?: boolean
  reasoningSteps?: string[]
}

export type Conversation = {
  threadId: string
  title: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
}

export type ChatRequest = {
  message: string
  threadId?: string
  userId?: string
  topK?: number
  includeReasoning?: boolean
}

export type ChatResponse = {
  answer: string
  thread_id: string
  sources: string[]
  confidence: number
  is_grounded: boolean
  route?: string | null
  reasoning_steps: string[]
}

export type ChatStreamEvent =
  | { type: 'start'; thread_id: string }
  | { type: 'token'; thread_id: string; content: string }
  | ({ type: 'final' } & ChatResponse)
  | { type: 'done'; thread_id: string }
  | { type: 'error'; thread_id?: string; message: string }

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { chatApi, getApiErrorMessage } from '../lib/api'
import type { ChatResponse, ChatStreamEvent } from '../lib/api'
import type { ChatMessage, Conversation } from '../types'

const STORAGE_KEY = 'tata-nexon-chat-ui-v1'
const DEFAULT_USER_ID = 'web-user'

type StoredChatState = {
  conversations: Conversation[]
  activeThreadId: string | null
}

type SendMessageOptions = {
  stream?: boolean
}

type StartedMessage = {
  assistantMessageId: string
  threadId: string
}

function loadStoredState(): StoredChatState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return { conversations: [], activeThreadId: null }
    }

    const parsed = JSON.parse(raw) as StoredChatState
    return {
      conversations: Array.isArray(parsed.conversations) ? parsed.conversations : [],
      activeThreadId: parsed.activeThreadId ?? null,
    }
  } catch {
    return { conversations: [], activeThreadId: null }
  }
}

function createId(prefix: string): string {
  if (crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function now(): string {
  return new Date().toISOString()
}

function titleFromMessage(message: string): string {
  const compact = message.trim().replace(/\s+/g, ' ')
  return compact.length > 42 ? `${compact.slice(0, 42)}...` : compact
}

function userMessage(content: string): ChatMessage {
  return {
    id: createId('msg'),
    role: 'user',
    content,
    createdAt: now(),
  }
}

function assistantPlaceholder(): ChatMessage {
  return {
    id: createId('msg'),
    role: 'assistant',
    content: '',
    createdAt: now(),
    pending: true,
  }
}

function updateAssistant(
  conversation: Conversation,
  assistantMessageId: string,
  updater: (message: ChatMessage) => ChatMessage,
): Conversation {
  return {
    ...conversation,
    updatedAt: now(),
    messages: conversation.messages.map((message) =>
      message.id === assistantMessageId ? updater(message) : message,
    ),
  }
}

export function useChat() {
  const initialState = useMemo(loadStoredState, [])
  const [conversations, setConversations] = useState<Conversation[]>(
    initialState.conversations,
  )
  const [activeThreadId, setActiveThreadId] = useState<string | null>(
    initialState.activeThreadId,
  )
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<EventSource | null>(null)

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.threadId === activeThreadId) ?? null,
    [activeThreadId, conversations],
  )

  const messages = activeConversation?.messages ?? []

  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ conversations, activeThreadId }),
    )
  }, [activeThreadId, conversations])

  useEffect(() => {
    return () => streamRef.current?.close()
  }, [])

  const updateConversation = useCallback(
    (threadId: string, updater: (conversation: Conversation) => Conversation) => {
      setConversations((current) =>
        current.map((conversation) =>
          conversation.threadId === threadId ? updater(conversation) : conversation,
        ),
      )
    },
    [],
  )

  const updateAssistantMessage = useCallback(
    (
      threadId: string,
      assistantMessageId: string,
      updater: (message: ChatMessage) => ChatMessage,
    ) => {
      updateConversation(threadId, (conversation) =>
        updateAssistant(conversation, assistantMessageId, updater),
      )
    },
    [updateConversation],
  )

  const newConversation = useCallback(() => {
    streamRef.current?.close()
    streamRef.current = null
    setIsLoading(false)
    setIsStreaming(false)
    setError(null)

    const timestamp = now()
    const conversation: Conversation = {
      threadId: createId('thread'),
      title: 'New chat',
      createdAt: timestamp,
      updatedAt: timestamp,
      messages: [],
    }

    setConversations((current) => [conversation, ...current])
    setActiveThreadId(conversation.threadId)
  }, [])

  const deleteConversation = useCallback(
    (threadId: string) => {
      setConversations((current) => current.filter((item) => item.threadId !== threadId))
      if (activeThreadId === threadId) {
        setActiveThreadId(null)
      }
    },
    [activeThreadId],
  )

  const startMessage = useCallback(
    (text: string): StartedMessage => {
      const threadId = activeThreadId ?? createId('thread')
      const timestamp = now()
      const outgoing = userMessage(text)
      const pendingAssistant = assistantPlaceholder()

      setError(null)
      setActiveThreadId(threadId)
      setConversations((current) => {
        const existing = current.find((conversation) => conversation.threadId === threadId)
        if (!existing) {
          return [
            {
              threadId,
              title: titleFromMessage(text),
              createdAt: timestamp,
              updatedAt: timestamp,
              messages: [outgoing, pendingAssistant],
            },
            ...current,
          ]
        }

        return current.map((conversation) =>
          conversation.threadId === threadId
            ? {
                ...conversation,
                title:
                  conversation.title === 'New chat'
                    ? titleFromMessage(text)
                    : conversation.title,
                updatedAt: timestamp,
                messages: [...conversation.messages, outgoing, pendingAssistant],
              }
            : conversation,
        )
      })

      return {
        assistantMessageId: pendingAssistant.id,
        threadId,
      }
    },
    [activeThreadId],
  )

  const finishRequest = useCallback(() => {
    streamRef.current = null
    setIsLoading(false)
    setIsStreaming(false)
  }, [])

  const applyFinalResponse = useCallback(
    (threadId: string, assistantMessageId: string, response: ChatResponse) => {
      updateAssistantMessage(threadId, assistantMessageId, (message) => ({
        ...message,
        content: response.answer,
        pending: false,
        sources: response.sources,
        confidence: response.confidence,
        isGrounded: response.is_grounded,
        reasoningSteps: response.reasoning_steps,
      }))
    },
    [updateAssistantMessage],
  )

  const applyAssistantError = useCallback(
    (threadId: string, assistantMessageId: string, message: string) => {
      setError(message)
      updateAssistantMessage(threadId, assistantMessageId, (chatMessage) => ({
        ...chatMessage,
        content: message,
        pending: false,
        error: true,
      }))
    },
    [updateAssistantMessage],
  )

  const sendNonStreamingMessage = useCallback(
    async (text: string, threadId: string, assistantMessageId: string) => {
      setIsLoading(true)
      try {
        const response = await chatApi.sendMessage({
          message: text,
          threadId,
          userId: DEFAULT_USER_ID,
          includeReasoning: true,
        })
        applyFinalResponse(threadId, assistantMessageId, response)
      } catch (requestError) {
        applyAssistantError(threadId, assistantMessageId, getApiErrorMessage(requestError))
      } finally {
        setIsLoading(false)
      }
    },
    [applyAssistantError, applyFinalResponse],
  )

  const handleStreamEvent = useCallback(
    (threadId: string, assistantMessageId: string, payload: ChatStreamEvent) => {
      if (payload.type === 'token') {
        updateAssistantMessage(threadId, assistantMessageId, (message) => ({
          ...message,
          content: `${message.content}${payload.content}`,
          pending: true,
        }))
        return
      }

      if (payload.type === 'final') {
        applyFinalResponse(threadId, assistantMessageId, payload)
        return
      }

      if (payload.type === 'done') {
        streamRef.current?.close()
        finishRequest()
        return
      }

      if (payload.type === 'error') {
        applyAssistantError(threadId, assistantMessageId, payload.message)
        streamRef.current?.close()
        finishRequest()
      }
    },
    [applyAssistantError, applyFinalResponse, finishRequest, updateAssistantMessage],
  )

  const sendStreamingMessage = useCallback(
    (text: string, threadId: string, assistantMessageId: string) => {
      setIsLoading(true)
      setIsStreaming(true)

      streamRef.current = chatApi.sendMessage(
        {
          message: text,
          threadId,
          userId: DEFAULT_USER_ID,
          includeReasoning: true,
        },
        {
          stream: true,
          onEvent: (event) => handleStreamEvent(threadId, assistantMessageId, event),
          onError: (message) => {
            applyAssistantError(threadId, assistantMessageId, message)
            finishRequest()
          },
        },
      )
    },
    [applyAssistantError, finishRequest, handleStreamEvent],
  )

  const sendMessage = useCallback(
    (content: string, options: SendMessageOptions = {}) => {
      const text = content.trim()
      if (!text || isLoading) {
        return
      }

      streamRef.current?.close()
      const { assistantMessageId, threadId } = startMessage(text)
      const shouldStream = options.stream ?? streamingEnabled

      if (shouldStream) {
        sendStreamingMessage(text, threadId, assistantMessageId)
      } else {
        void sendNonStreamingMessage(text, threadId, assistantMessageId)
      }
    },
    [
      isLoading,
      sendNonStreamingMessage,
      sendStreamingMessage,
      startMessage,
      streamingEnabled,
    ],
  )

  return {
    activeThreadId,
    conversations,
    error,
    isLoading,
    isStreaming,
    messages,
    streamingEnabled,
    deleteConversation,
    newConversation,
    selectConversation: setActiveThreadId,
    sendMessage,
    setStreamingEnabled,
  }
}

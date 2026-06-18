import axios from 'axios'

/**
 * Public shape accepted by the chat API client.
 *
 * The backend uses snake_case over HTTP, while the frontend keeps camelCase at
 * component boundaries. This client is the only place that translates between
 * the two.
 */
export interface ChatRequest {
  message: string
  threadId?: string
  userId?: string
  topK?: number
  includeReasoning?: boolean
}

/**
 * Stable response shape returned by the FastAPI `/chat` endpoint.
 */
export interface ChatResponse {
  answer: string
  thread_id: string
  sources: string[]
  confidence: number
  is_grounded: boolean
  route?: string | null
  reasoning_steps: string[]
}

export interface IngestOptions {
  forceReprocess?: boolean
  collectionName?: string
}

export interface IngestResponse {
  status: string
  source: string
  chunks_created: number
  chunks_stored: number
  metadata: Record<string, unknown>
}

export interface DocumentSummary {
  source: string
  chunks_created: number
  chunks_stored: number
  metadata: Record<string, unknown>
}

export interface DocumentsResponse {
  documents: DocumentSummary[]
}

/**
 * Server-Sent Event payloads emitted by the streaming chat endpoint.
 */
export type ChatStreamEvent =
  | { type: 'start'; thread_id: string }
  | { type: 'token'; thread_id: string; content: string }
  | ({ type: 'final' } & ChatResponse)
  | { type: 'done'; thread_id: string }
  | { type: 'error'; thread_id?: string; message: string }

export type StreamingChatOptions = {
  stream: true
  onEvent: (event: ChatStreamEvent) => void
  onError?: (message: string) => void
}

export type NonStreamingChatOptions = {
  stream?: false
}

type ApiErrorPayload = {
  detail?: unknown
  error?: {
    message?: unknown
  }
}

export const API_BASE_URL = (
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '')

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json',
  },
})

function requestPayload(request: ChatRequest) {
  return {
    message: request.message,
    thread_id: request.threadId,
    user_id: request.userId,
    top_k: request.topK,
    include_reasoning: request.includeReasoning ?? false,
  }
}

function streamingUrl(request: ChatRequest): string {
  const url = new URL('/chat', API_BASE_URL)
  url.searchParams.set('stream', 'true')
  url.searchParams.set('message', request.message)
  url.searchParams.set('include_reasoning', String(request.includeReasoning ?? true))

  if (request.threadId) {
    url.searchParams.set('thread_id', request.threadId)
  }
  if (request.userId) {
    url.searchParams.set('user_id', request.userId)
  }
  if (request.topK) {
    url.searchParams.set('top_k', String(request.topK))
  }

  return url.toString()
}

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiErrorPayload>(error)) {
    const detail = error.response?.data?.detail
    const nestedMessage = error.response?.data?.error?.message

    if (typeof detail === 'string') {
      return detail
    }
    if (typeof nestedMessage === 'string') {
      return nestedMessage
    }
    if (error.response?.status) {
      return `API returned ${error.response.status}.`
    }
    if (error.message) {
      return error.message
    }
  }

  if (error instanceof Error && error.message) {
    return error.message
  }

  return 'Something went wrong while contacting the chat API.'
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await api.post<ChatResponse>('/chat', requestPayload(request))
    return response.data
  } catch (error) {
    throw new Error(getApiErrorMessage(error))
  }
}

export async function ingestDocument(
  file: File,
  options: IngestOptions = {},
): Promise<IngestResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('source_filename', file.name)
  formData.append('force_reprocess', String(options.forceReprocess ?? false))

  if (options.collectionName?.trim()) {
    formData.append('collection_name', options.collectionName.trim())
  }

  try {
    const response = await api.post<IngestResponse>('/admin/ingest', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300_000,
    })
    return response.data
  } catch (error) {
    throw new Error(getApiErrorMessage(error))
  }
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  try {
    const response = await api.get<DocumentsResponse>('/admin/documents')
    return response.data.documents
  } catch (error) {
    throw new Error(getApiErrorMessage(error))
  }
}

export async function ingestDocuments(
  files: File[],
  options: IngestOptions = {},
  onProgress?: (completed: number, total: number, latest?: IngestResponse) => void,
): Promise<IngestResponse[]> {
  const results: IngestResponse[] = []

  for (const file of files) {
    const result = await ingestDocument(file, options)
    results.push(result)
    onProgress?.(results.length, files.length, result)
  }

  return results
}

export function createChatEventSource(
  request: ChatRequest,
  options?: Omit<StreamingChatOptions, 'stream'>,
): EventSource {
  const eventSource = new EventSource(streamingUrl(request))

  if (options) {
    eventSource.onmessage = (event) => {
      try {
        options.onEvent(JSON.parse(event.data) as ChatStreamEvent)
      } catch {
        options.onError?.('Invalid streaming response from chat API.')
        eventSource.close()
      }
    }

    eventSource.onerror = () => {
      options.onError?.('Could not connect to the chat API.')
      eventSource.close()
    }
  }

  return eventSource
}

function sendMessage(
  request: ChatRequest,
  options: StreamingChatOptions,
): EventSource
function sendMessage(
  request: ChatRequest,
  options?: NonStreamingChatOptions,
): Promise<ChatResponse>
function sendMessage(
  request: ChatRequest,
  options: StreamingChatOptions | NonStreamingChatOptions = {},
) {
  if (options.stream) {
    return createChatEventSource(request, {
      onEvent: options.onEvent,
      onError: options.onError,
    })
  }

  return sendChatMessage(request)
}

export const chatApi = {
  baseUrl: API_BASE_URL,
  sendMessage,
}

export const adminApi = {
  ingestDocument,
  ingestDocuments,
  listDocuments,
}

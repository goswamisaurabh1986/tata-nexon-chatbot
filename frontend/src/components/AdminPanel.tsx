import { type ChangeEvent, useEffect, useMemo, useState } from 'react'

import { adminApi, getApiErrorMessage } from '../lib/api'
import type { DocumentSummary, IngestResponse } from '../lib/api'

const supportedExtensions = ['.pdf', '.txt', '.md']

function formatMetadataValue(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return 'n/a'
  }
  return String(value)
}

function extensionLabel(): string {
  return supportedExtensions.join(', ').toUpperCase().replaceAll('.', '')
}

type AdminPanelProps = {
  isActive: boolean
}

export function AdminPanel({ isActive }: AdminPanelProps) {
  const [files, setFiles] = useState<File[]>([])
  const [forceReprocess, setForceReprocess] = useState(false)
  const [collectionName, setCollectionName] = useState('')
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [results, setResults] = useState<IngestResponse[]>([])
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false)
  const [isIngesting, setIsIngesting] = useState(false)
  const [completedFiles, setCompletedFiles] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const progressPercent = useMemo(() => {
    if (!files.length) {
      return 0
    }
    return Math.round((completedFiles / files.length) * 100)
  }, [completedFiles, files.length])

  const totalChunksCreated = useMemo(
    () => results.reduce((total, result) => total + result.chunks_created, 0),
    [results],
  )

  useEffect(() => {
    if (isActive) {
      void refreshDocuments()
    }
  }, [isActive])

  const refreshDocuments = async () => {
    setIsLoadingDocuments(true)
    setError(null)
    try {
      setDocuments(await adminApi.listDocuments())
    } catch (requestError) {
      setError(getApiErrorMessage(requestError))
    } finally {
      setIsLoadingDocuments(false)
    }
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFiles(Array.from(event.target.files ?? []))
    setResults([])
    setSuccess(null)
    setError(null)
    setCompletedFiles(0)
  }

  const ingestSelectedFiles = async () => {
    if (!files.length || isIngesting) {
      return
    }

    setIsIngesting(true)
    setCompletedFiles(0)
    setResults([])
    setSuccess(null)
    setError(null)

    try {
      const ingestResults = await adminApi.ingestDocuments(
        files,
        {
          forceReprocess,
          collectionName,
        },
        (completed) => setCompletedFiles(completed),
      )
      const chunksCreated = ingestResults.reduce(
        (total, result) => total + result.chunks_created,
        0,
      )

      setResults(ingestResults)
      setSuccess(
        `Ingested ${ingestResults.length} document${ingestResults.length === 1 ? '' : 's'} and created ${chunksCreated} chunks.`,
      )
      setDocuments(await adminApi.listDocuments())
    } catch (requestError) {
      setError(getApiErrorMessage(requestError))
    } finally {
      setIsIngesting(false)
    }
  }

  return (
    <main className="flex h-screen min-w-0 flex-1 flex-col bg-neutral-950">
      <header className="border-b border-neutral-800 bg-neutral-950/90 px-4 py-4 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-6xl">
          <p className="text-xs font-semibold uppercase text-cyan-300">Admin</p>
          <h1 className="text-xl font-semibold text-neutral-50">Ingestion Panel</h1>
          <p className="mt-1 text-sm text-neutral-500">
            Upload source documents and refresh the retrieval index.
          </p>
        </div>
      </header>

      <section className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-2xl shadow-black/20">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-neutral-50">Upload documents</h2>
                <p className="mt-1 text-sm text-neutral-400">
                  Supports {extensionLabel()} files. Multiple files are allowed.
                </p>
              </div>
              <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">
                {files.length} selected
              </span>
            </div>

            <label className="mt-5 flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-neutral-700 bg-neutral-950/70 px-5 py-8 text-center transition hover:border-cyan-400/70 hover:bg-neutral-950">
              <input
                accept={supportedExtensions.join(',')}
                className="sr-only"
                disabled={isIngesting}
                multiple
                onChange={handleFileChange}
                type="file"
              />
              <span className="text-sm font-semibold text-neutral-100">
                Choose files to ingest
              </span>
              <span className="mt-2 text-sm text-neutral-500">
                PDF, TXT, and MD files are sent to the admin ingestion endpoint.
              </span>
            </label>

            {files.length > 0 && (
              <div className="mt-4 rounded-xl border border-neutral-800 bg-neutral-950/60 p-3">
                <p className="mb-2 text-xs font-semibold uppercase text-neutral-500">
                  Selected files
                </p>
                <div className="space-y-2">
                  {files.map((file) => (
                    <div
                      className="flex items-center justify-between gap-3 text-sm"
                      key={`${file.name}-${file.lastModified}`}
                    >
                      <span className="truncate text-neutral-200">{file.name}</span>
                      <span className="shrink-0 text-xs text-neutral-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-5 grid gap-4 sm:grid-cols-[0.8fr_1.2fr]">
              <label className="flex items-center gap-3 rounded-xl border border-neutral-800 bg-neutral-950/50 px-4 py-3 text-sm text-neutral-200">
                <input
                  checked={forceReprocess}
                  className="h-4 w-4 accent-cyan-300"
                  disabled={isIngesting}
                  onChange={(event) => setForceReprocess(event.target.checked)}
                  type="checkbox"
                />
                Force Reprocess
              </label>

              <label className="block">
                <span className="mb-2 block text-xs font-semibold uppercase text-neutral-500">
                  Collection Name
                </span>
                <input
                  className="w-full rounded-xl border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-cyan-400/70"
                  disabled={isIngesting}
                  onChange={(event) => setCollectionName(event.target.value)}
                  placeholder="Optional"
                  type="text"
                  value={collectionName}
                />
              </label>
            </div>

            {isIngesting && (
              <div className="mt-5">
                <div className="mb-2 flex justify-between text-xs text-neutral-400">
                  <span>Ingesting documents</span>
                  <span>{progressPercent}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
                  <div
                    className="h-full rounded-full bg-cyan-300 transition-all"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="mt-5 rounded-xl border border-red-400/30 bg-red-950/40 px-4 py-3 text-sm text-red-100">
                {error}
              </div>
            )}

            {success && (
              <div className="mt-5 rounded-xl border border-emerald-400/30 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-100">
                {success}
              </div>
            )}

            <button
              className="mt-5 w-full rounded-xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-400"
              disabled={!files.length || isIngesting}
              onClick={ingestSelectedFiles}
              type="button"
            >
              {isIngesting ? 'Ingesting...' : 'Ingest Documents'}
            </button>

            {results.length > 0 && (
              <div className="mt-6">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-semibold text-neutral-100">Latest ingestion</h3>
                  <span className="rounded-full bg-neutral-950 px-3 py-1 text-xs text-neutral-400">
                    {totalChunksCreated} chunks
                  </span>
                </div>
                <div className="overflow-hidden rounded-xl border border-neutral-800">
                  {results.map((result) => (
                    <div
                      className="grid gap-2 border-b border-neutral-800 bg-neutral-950/50 p-4 text-sm last:border-b-0 sm:grid-cols-[1fr_auto]"
                      key={result.source}
                    >
                      <div>
                        <p className="font-medium text-neutral-100">{result.source}</p>
                        <p className="text-xs text-neutral-500">{result.status}</p>
                      </div>
                      <div className="text-left sm:text-right">
                        <p className="text-neutral-200">{result.chunks_created} chunks</p>
                        <p className="text-xs text-neutral-500">
                          {result.chunks_stored} stored
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-2xl shadow-black/20">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-neutral-50">
                  Ingested documents
                </h2>
                <p className="mt-1 text-sm text-neutral-400">
                  Current API process registry.
                </p>
              </div>
              <button
                className="rounded-xl border border-neutral-700 px-3 py-2 text-sm text-neutral-200 transition hover:border-cyan-400 hover:text-cyan-100 disabled:opacity-60"
                disabled={isLoadingDocuments}
                onClick={() => void refreshDocuments()}
                type="button"
              >
                {isLoadingDocuments ? 'Refreshing' : 'Refresh'}
              </button>
            </div>

            <div className="mt-5 space-y-3">
              {documents.length === 0 && !isLoadingDocuments ? (
                <div className="rounded-xl border border-neutral-800 bg-neutral-950/50 p-4 text-sm text-neutral-400">
                  No documents are registered yet.
                </div>
              ) : (
                documents.map((document) => (
                  <article
                    className="rounded-xl border border-neutral-800 bg-neutral-950/50 p-4"
                    key={document.source}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <h3 className="truncate font-medium text-neutral-100">
                          {document.source}
                        </h3>
                        <p className="mt-1 text-xs text-neutral-500">
                          {formatMetadataValue(document.metadata.file_type).toUpperCase()} |{' '}
                          {formatMetadataValue(document.metadata.page_count)} pages
                        </p>
                      </div>
                      <span className="shrink-0 rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">
                        {document.chunks_stored} chunks
                      </span>
                    </div>
                    {Boolean(document.metadata.collection_name) && (
                      <p className="mt-3 text-xs text-neutral-500">
                        Collection: {formatMetadataValue(document.metadata.collection_name)}
                      </p>
                    )}
                  </article>
                ))
              )}
            </div>
          </section>
        </div>
      </section>
    </main>
  )
}

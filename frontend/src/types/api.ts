export interface IngestResponse {
  success: boolean
  document_id: string | null
  total_chunks: number
  is_duplicate: boolean
  duplicate_of: string | null
  similarity_score: number
  total_duration_ms: number
  failed_stage: string | null
  error: string | null
}

export interface BatchIngestResponse {
  total: number
  succeeded: number
  failed: number
  duplicates: number
  results: IngestResponse[]
}

export interface ChunkResult {
  chunk_id: string
  document_id: string
  text: string
  score: number
  metadata: Record<string, unknown>
}

export interface QueryResponse {
  query: string
  chunks: ChunkResult[]
  window_texts: string[]
  retrieval_stats: Record<string, number>
  answer: string | null
  answer_model: string | null
}

export interface DocumentRecord {
  id: string
  file_path: string
  status: string
  tenant_id: string
  owner_id: string
  total_chunks: number
  failed_stage: string | null
  error: string | null
  is_duplicate: boolean
  duplicate_of: string | null
  similarity_score: number
  total_duration_ms: number
  created_at: string | null
  updated_at: string | null
}

export interface DocumentListResponse {
  documents: DocumentRecord[]
  total: number
  limit: number
  offset: number
}

export interface DeleteResponse {
  deleted: boolean
  document_id: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  chunks?: ChunkResult[]
  retrieval_stats?: Record<string, number>
  timestamp: Date
}

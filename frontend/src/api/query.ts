import { apiClient } from './client'
import type { QueryResponse } from '../types/api'

export async function queryDocuments(
  query: string,
  topK = 5,
  generateAnswer = true
): Promise<QueryResponse> {
  const { data } = await apiClient.post<QueryResponse>('/query', {
    query,
    top_k: topK,
    generate_answer: generateAnswer,
  })
  return data
}

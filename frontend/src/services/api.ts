import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface IndexingStatus {
  repo_id: string;
  status: 'processing' | 'completed' | 'failed';
  message: string;
  total_files: number;
  processed_files: number;
}

export interface CodeCitation {
  file_path: string;
  content: string;
  similarity_score: number;
  language: string | null;
  start_line: number | null;
}

export interface QueryResponse {
  answer: string;
  citations: CodeCitation[];
  repo_id: string;
  question: string;
  llm_mode: string;
}

export interface HealthCheck {
  status: string;
  ollama_connected: boolean;
  models_available: string[];
}

class APIService {
  private client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120000, // 2 minutes for LLM generation
  });

  async healthCheck(): Promise<HealthCheck> {
    const response = await this.client.get<HealthCheck>('/health');
    return response.data;
  }

  async ingestGithubRepo(githubUrl: string): Promise<IndexingStatus> {
    const response = await this.client.post<IndexingStatus>('/repository/github', {
      source_type: 'github',
      github_url: githubUrl,
    });
    return response.data;
  }

  async uploadRepository(file: File): Promise<IndexingStatus> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<IndexingStatus>(
      '/repository/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  async getIndexingStatus(repoId: string): Promise<IndexingStatus> {
    const response = await this.client.get<IndexingStatus>(
      `/repository/${repoId}/status`
    );
    return response.data;
  }

  async queryRepository(
    repoId: string,
    question: string,
    llmMode: 'ollama' | 'huggingface' = 'ollama'
  ): Promise<QueryResponse> {
    const response = await this.client.post<QueryResponse>('/query', {
      repo_id: repoId,
      question,
      llm_mode: llmMode,
    });
    return response.data;
  }

  async listRepositories(): Promise<{ repositories: IndexingStatus[] }> {
    const response = await this.client.get('/repositories');
    return response.data;
  }
}

export const apiService = new APIService();

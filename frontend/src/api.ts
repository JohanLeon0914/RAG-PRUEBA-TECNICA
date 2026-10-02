export type Source = {
  title: string;
  url: string;
  chunk_index: number | null;
  score: number | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  supported_by_context: boolean;
  sources: Source[];
  metadata: Record<string, unknown>;
};

export type ConversationSession = {
  id: string;
  created_at: string;
  updated_at: string;
};

export type ConversationMessage = {
  id: number;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  supported_by_context: boolean | null;
  sources_count: number | null;
};

export async function sendMessage(
  message: string,
  sessionId: string | null,
): Promise<ChatResponse> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail || 'El asistente no esta disponible.');
  }

  return response.json() as Promise<ChatResponse>;
}

export async function fetchSessions(): Promise<ConversationSession[]> {
  const response = await fetch('/api/sessions');
  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail || 'No se pudieron cargar las conversaciones.');
  }
  return response.json() as Promise<ConversationSession[]>;
}

export async function fetchSessionMessages(sessionId: string): Promise<ConversationMessage[]> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail || 'No se pudo cargar el historial.');
  }
  return response.json() as Promise<ConversationMessage[]>;
}

async function safeErrorDetail(response: Response): Promise<string | null> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? null;
  } catch {
    return null;
  }
}

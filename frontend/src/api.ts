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

async function safeErrorDetail(response: Response): Promise<string | null> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? null;
  } catch {
    return null;
  }
}

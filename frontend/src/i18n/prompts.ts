// Default agent prompts.
//
// SINGLE SOURCE OF TRUTH: the text lives in ./agent-prompts.json, which the
// Python backend also reads (backend/config/agent_prompts.py) to populate the
// defaults used by headless / API sessions. Edit the JSON, never a copy: the
// two used to be maintained separately and drifted apart in both directions —
// the frontend copy lost the organism-aware enrichment guidance and the
// anti-fabrication rules, the backend copy lost the honest-reporting blocks.
//
// These are deliberately English-only and do not follow the UI language.

import type { Language } from './translations'
import type { AgentPrompts } from '../stores/domain'
import defaults from './agent-prompts.json'

// Fixed English prompts - these do not change with UI language
export const DEFAULT_ENGLISH_PROMPTS: AgentPrompts = defaults

// Helper to get prompts - always returns English prompts (language-independent)
export function getDefaultPrompts(_lang: Language): AgentPrompts {
  return DEFAULT_ENGLISH_PROMPTS
}

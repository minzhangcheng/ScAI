import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Language, getDefaultPrompts } from '../i18n/translations'

// Types
export type AgentType = 'pi' | 'pathology' | 'bioinformatics' | 'biochemistry' | 'literature'
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'waiting_input'

export interface Message {
  id: string
  type: 'status' | 'message' | 'code' | 'result' | 'error' | 'figure' | 'step' | 'prompt' | 'response' | 'thinking' | 'tool_call' | 'output' | 'completed' | 'progress' | 'round_summary' | 'data_request' | 'human_input_received' | 'human_input_requested' | 'literature' | 'pi_question' | 'pi_response' | 'findings'
  agent?: AgentType
  content: any
  timestamp: string
}

export interface PathologyImage {
  id: string
  filename: string
  imageType: string
  preview?: string
}

export interface BiochemistryResult {
  id: string
  indicator: string
  value: number
  unit: string
}

export interface BiochemistryFile {
  id: string
  filename: string
  fileType: 'image' | 'file'
  description?: string
}

export interface RoundResult {
  round: number
  summary: string
  pathologyFindings: string[]
  bioinformaticsFindings: string[]
  biochemistryFindings: string[]
  literatureFindings: string[]
}

export interface Session {
  id: string
  name: string
  status: TaskStatus
  currentRound: number
  createdAt: string
  instruction: string
  pathologyImages: PathologyImage[]
  controlSeqPath?: string      // Control group sequencing data (.h5ad)
  treatmentSeqPath?: string    // Treatment group sequencing data (.h5ad)
  controlLoomPath?: string     // Control group loom file for RNA velocity
  treatmentLoomPath?: string   // Treatment group loom file for RNA velocity
  biochemistryResults: BiochemistryResult[]
  biochemistryFiles?: BiochemistryFile[]  // Uploaded files/images for biochemistry
  roundResults: RoundResult[]
  finalSummary?: string
}

export interface AgentPrompts {
  pi: string
  pathology: string
  bioinformatics: string
  biochemistry: string
  literature: string
}

export interface AgentModels {
  pi: string
  pathology: string
  bioinformatics: string
  biochemistry: string
  literature: string
}

export interface AgentTemperatures {
  pi: number
  pathology: number
  bioinformatics: number
  biochemistry: number
  literature: number
}

// Default models
export const DEFAULT_MODELS: AgentModels = {
  pi: 'google/gemini-3-pro-preview',
  pathology: 'google/gemini-3-pro-preview',
  bioinformatics: 'anthropic/claude-opus-4.5',
  biochemistry: 'google/gemini-3-pro-preview',
  literature: 'google/gemini-3-pro-preview'
}

// Default temperatures
// Low temperature (0.1-0.3): More deterministic, objective analysis
// High temperature (0.5-0.7): More creative, diverse exploration
export const DEFAULT_TEMPERATURES: AgentTemperatures = {
  pi: 0.5,           // PI can be more creative in synthesis and planning
  pathology: 0.2,    // Pathology needs objective, consistent analysis
  bioinformatics: 0.1, // Bioinformatics must be highly deterministic
  biochemistry: 0.2,  // Biochemistry needs objective interpretation
  literature: 0.6    // Literature search can explore more broadly
}

interface AppState {
  // Language
  language: Language
  setLanguage: (lang: Language) => void
  
  // API Config
  apiConfig: {
    apiKey: string
    googleApiKey: string
    llmBaseUrl: string
    proxyUrl: string
  }
  setApiConfig: (config: Partial<AppState['apiConfig']>) => void
  
  // Current session
  currentSession: Session | null
  setCurrentSession: (session: Session | null) => void
  updateSession: (updates: Partial<Session>) => void
  
  // Messages (real-time)
  messages: Message[]
  addMessage: (message: Message) => void
  setMessages: (messages: Message[]) => void
  clearMessages: () => void
  
  // Prompts
  prompts: AgentPrompts
  setPrompt: (agent: AgentType, prompt: string) => void
  resetPromptsToDefault: () => void
  getDefaultPromptsForLanguage: () => AgentPrompts
  
  // Models
  models: AgentModels
  setModel: (agent: AgentType, model: string) => void
  resetModelsToDefault: () => void
  
  // Temperatures
  temperatures: AgentTemperatures
  setTemperature: (agent: AgentType, temp: number) => void
  resetTemperaturesToDefault: () => void
  
  // Max Rounds
  maxRounds: number
  setMaxRounds: (rounds: number) => void
  
  // UI state
  activeTab: 'analysis' | 'data' | 'prompts' | 'results' | 'report'
  setActiveTab: (tab: AppState['activeTab']) => void
  
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void

  // Message panel width (percentage)
  messagePanelWidth: number
  setMessagePanelWidth: (width: number) => void

  // Report generation status
  reportGenerated: boolean
  setReportGenerated: (generated: boolean) => void

  // WebSocket
  wsConnected: boolean
  setWsConnected: (connected: boolean) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Language
      language: 'en' as Language,
      setLanguage: (lang) => {
        // Only update language, don't reset prompts (prompts are always English)
        set({ language: lang })
      },
      
      // API Config
      apiConfig: {
        apiKey: '',
        googleApiKey: '',
        llmBaseUrl: 'https://openrouter.ai/api/v1',
        proxyUrl: ''
      },
      setApiConfig: (config) => set((state) => ({
        apiConfig: { ...state.apiConfig, ...config }
      })),
      
      // Current session
      currentSession: null,
      setCurrentSession: (session) => set({ currentSession: session }),
      updateSession: (updates) => set((state) => ({
        currentSession: state.currentSession 
          ? { ...state.currentSession, ...updates }
          : null
      })),
      
      // Messages
      messages: [],
      addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
      })),
      setMessages: (messages) => set({ messages }),
      clearMessages: () => set({ messages: [] }),
      
      // Prompts
      prompts: getDefaultPrompts('en'),
      setPrompt: (agent, prompt) => set((state) => ({
        prompts: { ...state.prompts, [agent]: prompt }
      })),
      resetPromptsToDefault: () => set((state) => ({
        prompts: getDefaultPrompts(state.language)
      })),
      getDefaultPromptsForLanguage: () => {
        const state = get()
        return getDefaultPrompts(state.language)
      },
      
      // Models
      models: { ...DEFAULT_MODELS },
      setModel: (agent, model) => set((state) => ({
        models: { ...state.models, [agent]: model }
      })),
      resetModelsToDefault: () => set({ models: { ...DEFAULT_MODELS } }),
      
      // Temperatures
      temperatures: { ...DEFAULT_TEMPERATURES },
      setTemperature: (agent, temp) => set((state) => ({
        temperatures: { ...state.temperatures, [agent]: temp }
      })),
      resetTemperaturesToDefault: () => set({ temperatures: { ...DEFAULT_TEMPERATURES } }),
      
      // Max Rounds
      maxRounds: 3,
      setMaxRounds: (rounds) => set({ maxRounds: rounds }),
      
      // UI state - default to 'data' tab as users typically upload data first
      activeTab: 'data',
      setActiveTab: (tab) => set({ activeTab: tab }),
      
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      // Message panel width (percentage, 20-80)
      messagePanelWidth: 50,
      setMessagePanelWidth: (width) => set({ messagePanelWidth: Math.max(20, Math.min(80, width)) }),

      // Report generation status
      reportGenerated: false,
      setReportGenerated: (generated) => set({ reportGenerated: generated }),

      // WebSocket
      wsConnected: false,
      setWsConnected: (connected) => set({ wsConnected: connected })
    }),
    {
      name: 'intesc-storage',
      partialize: (state) => ({
        language: state.language,
        apiConfig: state.apiConfig,
        prompts: state.prompts,
        models: state.models,
        temperatures: state.temperatures,
        maxRounds: state.maxRounds,
        messagePanelWidth: state.messagePanelWidth,
        // Persist session and messages for page refresh recovery
        currentSession: state.currentSession,
        messages: state.messages.slice(-200), // Keep last 200 messages
      })
    }
  )
)

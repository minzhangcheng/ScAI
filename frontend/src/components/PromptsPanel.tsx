import { PromptEditor } from './prompts/PromptEditor'
import { usePromptsController } from './prompts/usePromptsController'

export function PromptsPanel() {
  return <PromptEditor controller={usePromptsController()} />
}

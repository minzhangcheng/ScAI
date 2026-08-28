import { PromptEditor } from './prompts/PromptEditor'
import { usePromptsController } from './prompts/usePromptsController'

export function PromptsPanel() {
  const controller = usePromptsController()
  return controller ? <PromptEditor controller={controller} /> : null
}

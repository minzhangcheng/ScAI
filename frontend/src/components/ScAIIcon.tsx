import type { ImgHTMLAttributes } from 'react'

type ScAIIconProps = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'>

export function ScAIIcon({ alt = '', draggable = false, ...props }: ScAIIconProps) {
  return (
    <img
      src="/scai-icon.png"
      alt={alt}
      draggable={draggable}
      {...props}
    />
  )
}

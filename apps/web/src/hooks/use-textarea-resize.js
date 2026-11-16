import { useEffect, useRef } from "react";

export function useTextareaResize(value, rows = 1) {
  const textareaRef = useRef(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    const computed = window.getComputedStyle(textarea);
    const lineHeight = parseInt(computed.lineHeight, 10) || 20;
    const minHeight = lineHeight * rows;

    const newHeight = Math.max(textarea.scrollHeight, minHeight);
    textarea.style.height = `${newHeight}px`;
  }, [value, rows]);

  return textareaRef;
}

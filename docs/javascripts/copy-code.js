(() => {
  const PROMPT_REGEX = /^(>>> |\.\.\. )/;

  const normalizeLines = (text) =>
    text
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.replace(PROMPT_REGEX, ""))
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trimEnd();

  const findCodeElement = (node) => {
    if (!node) {
      return null;
    }
    if (node.nodeType === Node.ELEMENT_NODE) {
      const el = node;
      if (el.matches("pre > code")) {
        return el;
      }
      return el.closest("pre")?.querySelector("code") || null;
    }
    return node.parentElement?.closest("pre")?.querySelector("code") || null;
  };

  const copyText = (text) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  };

  const handleCopyButton = (event) => {
    const button = event.target.closest(
      "button.md-clipboard, [data-clipboard-target]"
    );
    if (!button) {
      return;
    }
    const container = button.closest(
      "div.highlight, div.codehilite, figure.highlight, figure.codehilite"
    );
    const code = container?.querySelector("pre > code");
    if (!code) {
      return;
    }
    const original = code.textContent || "";
    const cleaned = normalizeLines(original);
    if (cleaned === original) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (typeof event.stopImmediatePropagation === "function") {
      event.stopImmediatePropagation();
    }
    copyText(cleaned);
  };

  const handleCopySelection = (event) => {
    const selection = document.getSelection();
    if (!selection || selection.isCollapsed) {
      return;
    }
    const code = findCodeElement(selection.anchorNode);
    if (!code) {
      return;
    }
    const selectedText = selection.toString();
    const cleaned = normalizeLines(selectedText);
    if (cleaned === selectedText) {
      return;
    }
    event.preventDefault();
    if (event.clipboardData) {
      event.clipboardData.setData("text/plain", cleaned);
    } else {
      copyText(cleaned);
    }
  };

  document.addEventListener("click", handleCopyButton, true);
  document.addEventListener("copy", handleCopySelection);
})();

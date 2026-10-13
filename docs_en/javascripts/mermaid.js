mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "base",
  themeVariables: {
    background: "transparent",
    primaryColor: "#e8f0f8",
    primaryTextColor: "#10243e",
    primaryBorderColor: "#1877a8",
    lineColor: "#66788c",
    secondaryColor: "#fff3e8",
    tertiaryColor: "#f7f8fa",
    fontFamily: "inherit",
  },
});

let diagramSequence = 0;

async function renderMermaidDiagrams() {
  const sources = document.querySelectorAll("pre.cf-mermaid");

  for (const source of sources) {
    const definition = source.textContent.trim();
    const id = `cf-mermaid-${diagramSequence++}`;

    try {
      const { svg, bindFunctions } = await mermaid.render(id, definition);
      const rendered = document.createElement("div");
      rendered.className = "cf-mermaid";
      rendered.innerHTML = svg;
      source.replaceWith(rendered);
      bindFunctions?.(rendered);
    } catch (error) {
      source.classList.add("cf-mermaid--error");
      console.error("Unable to render Mermaid diagram", error);
    }
  }
}

document$.subscribe(() => {
  renderMermaidDiagrams();
});

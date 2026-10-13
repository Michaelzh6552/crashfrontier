(() => {
  const storageKey = "crashfrontier-color-scheme";
  const windowPrefix = "crashfrontier-theme:";

  function readScheme() {
    if (window.name.startsWith(windowPrefix)) {
      return window.name.slice(windowPrefix.length);
    }

    try {
      return localStorage.getItem(storageKey);
    } catch {
      return null;
    }
  }

  function writeScheme(scheme) {
    window.name = `${windowPrefix}${scheme}`;

    try {
      localStorage.setItem(storageKey, scheme);
    } catch {
      // window.name keeps the choice when file:// storage is unavailable.
    }
  }

  function applyScheme(scheme) {
    const input = document.querySelector(
      `input[data-md-color-scheme="${scheme}"]`
    );

    if (!input) {
      return;
    }

    input.checked = true;
    document
      .querySelectorAll('form[data-md-component="palette"] label')
      .forEach((label) => {
        label.hidden = label.htmlFor === input.id;
      });
    if (document.body.dataset.mdColorScheme !== scheme) {
      document.body.setAttribute("data-md-color-scheme", scheme);
    }
    if (input.dataset.mdColorPrimary) {
      document.body.setAttribute(
        "data-md-color-primary",
        input.dataset.mdColorPrimary
      );
    }
    if (input.dataset.mdColorAccent) {
      document.body.setAttribute(
        "data-md-color-accent",
        input.dataset.mdColorAccent
      );
    }
  }

  const stored = readScheme();
  const initial =
    stored === "default" || stored === "slate"
      ? stored
      : matchMedia("(prefers-color-scheme: dark)").matches
        ? "slate"
        : "default";

  applyScheme(initial);
  writeScheme(initial);

  new MutationObserver(() => {
    const scheme = document.body.dataset.mdColorScheme;

    if (scheme === "default" || scheme === "slate") {
      writeScheme(scheme);
      applyScheme(scheme);
    }
  }).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });

  document.addEventListener("change", (event) => {
    const input = event.target.closest?.("input[data-md-color-scheme]");

    if (!input) {
      return;
    }

    const scheme = input.dataset.mdColorScheme;
    if (scheme !== "default" && scheme !== "slate") {
      return;
    }
    writeScheme(scheme);
    applyScheme(scheme);
  });
})();

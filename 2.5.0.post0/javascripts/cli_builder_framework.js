/**
 * Reusable terminal-style CLI command builder framework.
 * Use this as a base and provide command-specific configuration.
 */
(function registerTrackersCLIBuilderFramework(global) {
  "use strict";

  if (global.TrackersCLIBuilderFramework) {
    return;
  }

  class TerminalCommandBuilder {
    constructor(options) {
      this.root = options.root;
      this.defaults = { ...(options.defaults || {}) };
      this.numericFields = { ...(options.numericFields || {}) };
      this.state = { ...(options.initialState || {}) };
      this.generateCommand = options.generateCommand || (() => "");
      this.getValidationErrors = options.getValidationErrors || (() => []);
      this.renderBody = options.renderBody || (() => {});
      this.inputSanitizers = options.inputSanitizers || {};
      this.onInputChange = options.onInputChange || null;
      this.onCheckboxChange = options.onCheckboxChange || null;
      this.onSelectorChange = options.onSelectorChange || null;
      this.copyButtonFeedbackMs = options.copyButtonFeedbackMs || 1200;

      this.ids = {
        command: "cb-command",
        errors: "cb-errors",
        ...(options.ids || {}),
      };
      this.containerClassName = options.containerClassName || "cb-container";

      this.commandEl = null;
      this.errorsEl = null;

      this.numericHoldTimeout = null;
      this.numericHoldInterval = null;
      this.activeNumericHoldButton = null;
      this.eventsBound = false;

      this.boundHandlePointerDown = this.handlePointerDown.bind(this);
      this.boundHandleClick = this.handleClick.bind(this);
      this.boundHandleChange = this.handleChange.bind(this);
      this.boundHandleInput = this.handleInput.bind(this);
      this.boundClearNumericHold = this.clearNumericHold.bind(this);
      this.boundVisibilityChange = this.handleVisibilityChange.bind(this);
      this.boundApplyNativeMetrics = this.applyNativeMetrics.bind(this);
    }

    init() {
      if (!this.root) return false;
      this.render();
      this.bindEvents();
      this.syncNativeMetricsWithRetries();
      window.addEventListener("resize", this.boundApplyNativeMetrics);
      return true;
    }

    destroy() {
      this.unbindEvents();
      window.removeEventListener("resize", this.boundApplyNativeMetrics);
      this.clearNumericHold();
      this.root = null;
      this.commandEl = null;
      this.errorsEl = null;
    }

    render() {
      if (!this.root) return;

      this.root.innerHTML = "";
      this.root.className = this.containerClassName;

      const codeBlock = document.createElement("div");
      codeBlock.className = "cb-code-block language-text highlight";

      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.className = "cb-code";

      this.renderBody(code, this.getRenderApi());

      const errors = document.createElement("div");
      errors.id = this.ids.errors;
      errors.className = "cb-errors";
      code.appendChild(errors);

      pre.appendChild(code);
      codeBlock.appendChild(pre);
      this.root.appendChild(codeBlock);

      const output = this.createCommandOutputBlock();
      this.root.appendChild(output.block);

      this.commandEl = output.commandCode;
      this.errorsEl = errors;

      this.refresh();
    }

    getRenderApi() {
      return {
        ids: this.ids,
        defaults: this.defaults,
        numericFields: this.numericFields,
        state: this.state,
        root: this.root,
        createHeader: this.createHeader.bind(this),
        createSelector: this.createSelector.bind(this),
        createCheckbox: this.createCheckbox.bind(this),
        createTextInputRow: this.createTextInputRow.bind(this),
        createNumericInputRow: this.createNumericInputRow.bind(this),
        createSelectorRow: this.createSelectorRow.bind(this),
      };
    }

    createCommandOutputBlock() {
      const commandBlock = document.createElement("div");
      commandBlock.className = "language-text highlight cb-output";

      const commandPre = document.createElement("pre");
      commandPre.appendChild(document.createElement("span"));

      const commandCode = document.createElement("code");
      commandCode.id = this.ids.command;
      commandPre.appendChild(commandCode);
      commandBlock.appendChild(commandPre);

      const commandNav = document.createElement("nav");
      commandNav.className = "md-code__nav";

      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "md-code__button";
      copyButton.title = "Copy to clipboard";
      copyButton.setAttribute("aria-label", "Copy to clipboard");
      copyButton.setAttribute("data-md-type", "copy");
      copyButton.setAttribute("data-clipboard-target", `#${this.ids.command}`);
      copyButton.addEventListener("click", async () => {
        // Fallback clipboard handling for dynamically-rendered blocks.
        const commandText = (this.commandEl?.textContent || "").replace(/\\\n\s+/g, " ");
        if (!commandText) return;
        try {
          await navigator.clipboard.writeText(commandText);
          copyButton.classList.add("md-code__button--active");
          window.setTimeout(() => {
            copyButton.classList.remove("md-code__button--active");
          }, this.copyButtonFeedbackMs);
        } catch (_error) {
          // Native Material copy handler remains available when present.
        }
      });

      commandNav.appendChild(copyButton);
      commandBlock.appendChild(commandNav);

      return { block: commandBlock, commandCode };
    }

    refresh() {
      if (this.commandEl) {
        this.commandEl.textContent = this.generateCommand(this.state, this);
      }

      if (!this.errorsEl) return;

      const errors = this.getValidationErrors(this.state, this);
      if (errors.length > 0) {
        this.errorsEl.innerHTML = errors
          .map((errorText) => `<span class="cb-error"># ${errorText}</span>`)
          .join("\n");
        this.errorsEl.style.display = "block";
      } else {
        this.errorsEl.style.display = "none";
      }
    }

    bindEvents() {
      if (this.eventsBound || !this.root) return;

      this.root.addEventListener("pointerdown", this.boundHandlePointerDown);
      this.root.addEventListener("click", this.boundHandleClick);
      this.root.addEventListener("change", this.boundHandleChange);
      this.root.addEventListener("input", this.boundHandleInput);

      document.addEventListener("pointerup", this.boundClearNumericHold);
      document.addEventListener("pointercancel", this.boundClearNumericHold);
      window.addEventListener("blur", this.boundClearNumericHold);
      document.addEventListener("visibilitychange", this.boundVisibilityChange);

      this.eventsBound = true;
    }

    unbindEvents() {
      if (!this.eventsBound || !this.root) return;

      this.root.removeEventListener("pointerdown", this.boundHandlePointerDown);
      this.root.removeEventListener("click", this.boundHandleClick);
      this.root.removeEventListener("change", this.boundHandleChange);
      this.root.removeEventListener("input", this.boundHandleInput);

      document.removeEventListener("pointerup", this.boundClearNumericHold);
      document.removeEventListener("pointercancel", this.boundClearNumericHold);
      window.removeEventListener("blur", this.boundClearNumericHold);
      document.removeEventListener("visibilitychange", this.boundVisibilityChange);

      this.eventsBound = false;
    }

    handleVisibilityChange() {
      if (document.hidden) {
        this.clearNumericHold();
      }
    }

    handlePointerDown(event) {
      const numericButton = event.target.closest(".cb-num-btn");
      if (!numericButton) return;
      if (event.pointerType === "mouse" && event.button !== 0) return;

      event.preventDefault();
      this.startNumericHold(numericButton);
    }

    handleClick(event) {
      const selectorButton = event.target.closest(".cb-selector");
      if (selectorButton) {
        this.handleSelectorSelection(selectorButton);
        return;
      }

      const numericButton = event.target.closest(".cb-num-btn");
      if (!numericButton) return;

      // Pointer interaction is handled by pointerdown to support hold-repeat.
      // Keep click handling for keyboard activation.
      if (event.detail !== 0) return;
      this.stepNumericField(numericButton.dataset.key, numericButton.dataset.action);
    }

    handleSelectorSelection(selectorButton) {
      const group = selectorButton.dataset.group;
      const value = selectorButton.dataset.value;
      if (!group) return;

      this.state[group] = value;
      this.updateSelectorGroup(group, value);

      if (typeof this.onSelectorChange === "function") {
        this.onSelectorChange({
          group,
          value,
          button: selectorButton,
          builder: this,
          state: this.state,
        });
      }

      this.refresh();
    }

    updateSelectorGroup(group, selectedValue) {
      this.root.querySelectorAll(`.cb-selector[data-group="${group}"]`).forEach((button) => {
        const isActive = button.dataset.value === selectedValue;
        button.classList.toggle("cb-selector--active", isActive);
      });
    }

    handleChange(event) {
      const input = event.target;
      const key = input?.dataset?.key;
      if (!key) return;

      if (input.type === "checkbox") {
        this.processCheckboxInput(input);
        return;
      }

      this.processValueInput(input, { isCommit: true });
    }

    handleInput(event) {
      const input = event.target;
      const key = input?.dataset?.key;
      if (!key || input.type === "checkbox") return;

      this.processValueInput(input, { isCommit: false });
    }

    processCheckboxInput(input) {
      const key = input.dataset.key;
      const checked = input.checked;

      this.state[key] = checked;

      const checkbox = input.closest(".cb-checkbox");
      if (checkbox) {
        this.updateCheckboxVisual(checkbox, checked);
      }

      this.toggleCollapsibleForKey(key, checked);

      if (typeof this.onCheckboxChange === "function") {
        this.onCheckboxChange({ key, checked, input, builder: this, state: this.state });
      }

      this.refresh();
    }

    processValueInput(input, options) {
      const key = input.dataset.key;
      const isCommit = Boolean(options?.isCommit);

      const sanitizedValue = this.sanitizeInputValue(key, input.value, input, isCommit);
      if (sanitizedValue !== input.value) {
        input.value = sanitizedValue;
      }

      this.state[key] = input.value;

      if (input.classList.contains("cb-num-input")) {
        this.updateNumericModifiedState(input);
      }

      if (typeof this.onInputChange === "function") {
        this.onInputChange({
          key,
          value: input.value,
          input,
          isCommit,
          builder: this,
          state: this.state,
        });
      }

      this.refresh();
    }

    sanitizeInputValue(key, value, input, isCommit) {
      const sanitizer = this.inputSanitizers[key];
      if (typeof sanitizer !== "function") {
        return value;
      }
      return sanitizer(value, { key, input, isCommit, state: this.state, builder: this });
    }

    updateCheckboxVisual(checkbox, checked) {
      const marker = checkbox.querySelector(".cb-check-marker");
      if (marker) {
        marker.textContent = checked ? "[x]" : "[ ]";
      }
      checkbox.classList.toggle("cb-checkbox--active", checked);
    }

    toggleCollapsibleForKey(key, visible) {
      const collapsible = this.root.querySelector(`[data-collapsible-for="${key}"]`);
      if (collapsible) {
        collapsible.style.display = visible ? "block" : "none";
      }
    }

    setCheckboxField(key, checked, options = {}) {
      const shouldRefresh = options.refresh !== false;
      this.state[key] = checked;

      const input = this.root.querySelector(`input[type="checkbox"][data-key="${key}"]`);
      if (input) {
        input.checked = checked;
        const checkbox = input.closest(".cb-checkbox");
        if (checkbox) {
          this.updateCheckboxVisual(checkbox, checked);
        }
      }

      this.toggleCollapsibleForKey(key, checked);

      if (shouldRefresh) {
        this.refresh();
      }
    }

    setElementVisibleById(id, visible) {
      const target = this.root.querySelector(`#${id}`);
      if (target) {
        target.style.display = visible ? "block" : "none";
      }
    }

    getNumericInputForKey(key) {
      return this.root.querySelector(`input[data-key="${key}"].cb-num-input`);
    }

    updateNumericModifiedState(input) {
      const isModified = input.value !== input.dataset.default;
      input.classList.toggle("cb-modified", isModified);
    }

    stepNumericField(key, action) {
      const input = this.getNumericInputForKey(key);
      if (!input) return;

      const step = parseFloat(input.dataset.step);
      const min = parseFloat(input.dataset.min);
      const max = parseFloat(input.dataset.max);
      const decimals = parseInt(input.dataset.decimals, 10);

      let value = parseFloat(input.value);
      if (Number.isNaN(value)) {
        value = parseFloat(input.dataset.default);
      }
      if (Number.isNaN(value)) {
        value = min;
      }

      if (action === "increment") {
        value = Math.min(max, value + step);
      } else {
        value = Math.max(min, value - step);
      }

      const nextValue = decimals === 0 ? String(Math.round(value)) : value.toFixed(decimals);

      this.state[key] = nextValue;
      input.value = nextValue;
      this.updateNumericModifiedState(input);

      this.refresh();
    }

    startNumericHold(button) {
      this.clearNumericHold();

      this.activeNumericHoldButton = button;
      this.stepNumericField(button.dataset.key, button.dataset.action);

      this.numericHoldTimeout = window.setTimeout(() => {
        this.numericHoldInterval = window.setInterval(() => {
          if (!this.activeNumericHoldButton || !this.root.contains(this.activeNumericHoldButton)) {
            this.clearNumericHold();
            return;
          }

          this.stepNumericField(
            this.activeNumericHoldButton.dataset.key,
            this.activeNumericHoldButton.dataset.action
          );
        }, 90);
      }, 350);
    }

    clearNumericHold() {
      if (this.numericHoldTimeout !== null) {
        window.clearTimeout(this.numericHoldTimeout);
        this.numericHoldTimeout = null;
      }
      if (this.numericHoldInterval !== null) {
        window.clearInterval(this.numericHoldInterval);
        this.numericHoldInterval = null;
      }
      this.activeNumericHoldButton = null;
    }

    // Sync sizing metrics from a native MkDocs code block/button.
    applyNativeMetrics() {
      if (!this.root) return;

      const refCode =
        document.querySelector(".tabbed-block .language-text.highlight pre code") ||
        document.querySelector(".language-text.highlight pre code");
      if (refCode) {
        const codeStyles = window.getComputedStyle(refCode);
        this.root.style.setProperty("--cb-native-code-font-size", codeStyles.fontSize);
        this.root.style.setProperty("--cb-native-code-line-height", codeStyles.lineHeight);
        this.root.style.setProperty("--cb-native-code-font-family", codeStyles.fontFamily);
      }

      const refButton =
        document.querySelector(".tabbed-block .md-code__button[data-md-type='copy']") ||
        document.querySelector(".md-code__button[data-md-type='copy']");
      if (refButton) {
        const buttonStyles = window.getComputedStyle(refButton);
        this.root.style.setProperty("--cb-native-copy-button-width", buttonStyles.width);
        this.root.style.setProperty("--cb-native-copy-button-height", buttonStyles.height);
        this.root.style.setProperty("--cb-native-copy-button-font-size", buttonStyles.fontSize);
      }

      const refNav =
        document.querySelector(".tabbed-block .md-code__nav") || document.querySelector(".md-code__nav");
      if (refNav) {
        const navStyles = window.getComputedStyle(refNav);
        this.root.style.setProperty("--cb-native-copy-nav-padding", navStyles.padding);
        this.root.style.setProperty("--cb-native-copy-nav-right", navStyles.right);
        this.root.style.setProperty("--cb-native-copy-nav-top", navStyles.top);
        this.root.style.setProperty("--cb-native-copy-nav-border-radius", navStyles.borderRadius);
      }
    }

    syncNativeMetricsWithRetries() {
      let attempts = 0;
      const maxAttempts = 30;

      const tick = () => {
        this.applyNativeMetrics();
        attempts += 1;
        if (attempts < maxAttempts) {
          window.setTimeout(tick, 100);
        }
      };

      tick();
    }

    createSelector(label, value, group, isSelected) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `cb-selector ${isSelected ? "cb-selector--active" : ""}`;
      button.textContent = `[ ${label} ]`;
      button.dataset.value = value;
      button.dataset.group = group;
      return button;
    }

    createCheckbox(label, key, checked) {
      const wrapper = document.createElement("label");
      wrapper.className = `cb-checkbox ${checked ? "cb-checkbox--active" : ""}`;

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = checked;
      input.dataset.key = key;

      const marker = document.createElement("span");
      marker.className = "cb-check-marker";
      marker.textContent = checked ? "[x]" : "[ ]";

      const text = document.createElement("span");
      text.className = "cb-check-label";
      text.textContent = label;

      wrapper.appendChild(input);
      wrapper.appendChild(marker);
      wrapper.appendChild(text);
      return wrapper;
    }

    createHeader(title) {
      const header = document.createElement("div");
      header.className = "cb-header";
      header.textContent = `# ${title.toUpperCase()}`;
      return header;
    }

    createTextInputRow(label, placeholder, key, value, rowClass = "") {
      const row = document.createElement("div");
      row.className = `cb-row ${rowClass}`.trim();

      const labelEl = document.createElement("span");
      labelEl.className = "cb-label";
      labelEl.textContent = label;

      const equals = document.createElement("span");
      equals.className = "cb-equals";
      equals.textContent = "=";

      const valueWrap = document.createElement("span");
      valueWrap.className = "cb-value-wrap cb-value-wrap--text";

      const leftBracket = document.createElement("span");
      leftBracket.className = "cb-bracket";
      leftBracket.textContent = "[";

      const input = document.createElement("input");
      input.type = "text";
      input.className = "cb-text-input";
      input.placeholder = placeholder;
      input.value = value;
      input.dataset.key = key;
      input.dataset.placeholder = placeholder;

      const rightBracket = document.createElement("span");
      rightBracket.className = "cb-bracket";
      rightBracket.textContent = "]";

      valueWrap.appendChild(leftBracket);
      valueWrap.appendChild(input);
      valueWrap.appendChild(rightBracket);

      row.appendChild(labelEl);
      row.appendChild(equals);
      row.appendChild(valueWrap);

      return row;
    }

    createNumericInputRow(label, key, value, config, rowClass = "") {
      const row = document.createElement("div");
      row.className = `cb-row ${rowClass}`.trim();

      const labelEl = document.createElement("span");
      labelEl.className = "cb-label";
      labelEl.textContent = label;

      const equals = document.createElement("span");
      equals.className = "cb-equals";
      equals.textContent = "=";

      const controls = document.createElement("div");
      controls.className = "cb-numeric-controls";

      const decButton = document.createElement("button");
      decButton.type = "button";
      decButton.className = "cb-num-btn";
      decButton.textContent = "[-]";
      decButton.dataset.action = "decrement";
      decButton.dataset.key = key;

      const valueWrap = document.createElement("span");
      valueWrap.className = "cb-value-wrap cb-value-wrap--numeric";

      const leftBracket = document.createElement("span");
      leftBracket.className = "cb-bracket";
      leftBracket.textContent = "[";

      const input = document.createElement("input");
      input.type = "text";
      input.className = "cb-num-input";
      input.value = value;
      input.dataset.key = key;
      input.dataset.step = config.step;
      input.dataset.min = config.min;
      input.dataset.max = config.max;
      input.dataset.decimals = config.decimals;
      input.dataset.default = Object.prototype.hasOwnProperty.call(this.defaults, key)
        ? this.defaults[key]
        : value;

      const rightBracket = document.createElement("span");
      rightBracket.className = "cb-bracket";
      rightBracket.textContent = "]";

      const incButton = document.createElement("button");
      incButton.type = "button";
      incButton.className = "cb-num-btn";
      incButton.textContent = "[+]";
      incButton.dataset.action = "increment";
      incButton.dataset.key = key;

      valueWrap.appendChild(leftBracket);
      valueWrap.appendChild(input);
      valueWrap.appendChild(rightBracket);

      controls.appendChild(decButton);
      controls.appendChild(valueWrap);
      controls.appendChild(incButton);

      row.appendChild(labelEl);
      row.appendChild(equals);
      row.appendChild(controls);

      return row;
    }

    createSelectorRow(label, options, group, selectedValue, rowClass = "") {
      const row = document.createElement("div");
      row.className = `cb-row ${rowClass}`.trim();

      const labelEl = document.createElement("span");
      labelEl.className = "cb-label";
      labelEl.textContent = label;

      const equals = document.createElement("span");
      equals.className = "cb-equals";
      equals.textContent = "=";

      const selectors = document.createElement("div");
      selectors.className = "cb-selectors";

      options.forEach((option) => {
        selectors.appendChild(
          this.createSelector(option, option, group, option === selectedValue)
        );
      });

      row.appendChild(labelEl);
      row.appendChild(equals);
      row.appendChild(selectors);

      return row;
    }
  }

  global.TrackersCLIBuilderFramework = {
    TerminalCommandBuilder,
  };
})(window);

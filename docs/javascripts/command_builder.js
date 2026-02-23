/**
 * Track command widget built on top of the reusable CLI builder framework.
 */
document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("command-builder-widget");
  const framework = window.TrackersCLIBuilderFramework;
  if (!root || !framework || !framework.TerminalCommandBuilder) return;

  // Default values for reference
  const defaults = {
    source: "source.mp4",
    confidence: "0.5",
    classes: "",
    lostTrackBuffer: "30",
    trackActivationThreshold: "0.25",
    minimumConsecutiveFrames: "3",
    minimumIouThreshold: "0.3",
    output: "",
  };

  // Field configurations for numeric inputs
  const numericFields = {
    confidence: { step: 0.05, min: 0.05, max: 1, decimals: 2 },
    trackActivationThreshold: { step: 0.05, min: 0.05, max: 1, decimals: 2 },
    minimumIouThreshold: { step: 0.05, min: 0.05, max: 1, decimals: 2 },
    lostTrackBuffer: { step: 1, min: 1, max: 999, decimals: 0 },
    minimumConsecutiveFrames: { step: 1, min: 1, max: 99, decimals: 0 },
  };

  // State
  const initialState = {
    source: "",
    modelType: "detection",
    modelSize: "nano",
    confidence: "0.5",
    device: "auto",
    classes: "",
    tracker: "bytetrack",
    lostTrackBuffer: "30",
    trackActivationThreshold: "0.25",
    minimumConsecutiveFrames: "3",
    minimumIouThreshold: "0.3",
    display: false,
    showBoxes: true,
    showMasks: false,
    showConfidence: false,
    showLabels: false,
    showIds: true,
    showTrajectories: false,
    output: "",
    overwrite: false,
    showModelOptions: false,
    showTrackerOptions: false,
  };

  // Validation functions
  function isValidDecimal01(value, min = 0) {
    if (value === "") return true;
    const num = parseFloat(value);
    return !isNaN(num) && num >= min && num <= 1;
  }

  function isValidPositiveInt(value) {
    if (value === "") return true;
    const num = parseInt(value, 10);
    return !isNaN(num) && num > 0 && String(num) === value;
  }

  function isValidClasses(value) {
    return /^[\w,\s]*$/.test(value);
  }

  // Generate command from state
  function generateCommand(state) {
    const parts = ["trackers track"];

    const sourceValue = state.source.trim() || defaults.source;
    if (sourceValue) {
      parts.push(`--source ${sourceValue}`);
    }

    const prefix = state.modelType === "segmentation" ? "rfdetr-seg-" : "rfdetr-";
    parts.push(`--model ${prefix}${state.modelSize}`);

    if (state.showModelOptions) {
      if (state.confidence !== defaults.confidence && isValidDecimal01(state.confidence, 0.05)) {
        parts.push(`--model_confidence ${state.confidence}`);
      }
      if (state.device !== "auto") {
        parts.push(`--model_device ${state.device}`);
      }
      if (state.classes && isValidClasses(state.classes)) {
        parts.push(`--classes ${state.classes}`);
      }
    }

    parts.push(`--tracker ${state.tracker}`);

    const trackerParams = {};
    if (state.showTrackerOptions) {
      if (state.lostTrackBuffer !== defaults.lostTrackBuffer && isValidPositiveInt(state.lostTrackBuffer)) {
        trackerParams.lost_track_buffer = parseInt(state.lostTrackBuffer, 10);
      }
      if (
        state.trackActivationThreshold !== defaults.trackActivationThreshold &&
        isValidDecimal01(state.trackActivationThreshold, 0.05)
      ) {
        trackerParams.track_activation_threshold = parseFloat(state.trackActivationThreshold);
      }
      if (
        state.minimumConsecutiveFrames !== defaults.minimumConsecutiveFrames &&
        isValidPositiveInt(state.minimumConsecutiveFrames)
      ) {
        trackerParams.minimum_consecutive_frames = parseInt(state.minimumConsecutiveFrames, 10);
      }
      if (
        state.minimumIouThreshold !== defaults.minimumIouThreshold &&
        isValidDecimal01(state.minimumIouThreshold, 0.05)
      ) {
        trackerParams.minimum_iou_threshold = parseFloat(state.minimumIouThreshold);
      }
    }
    if (Object.keys(trackerParams).length > 0) {
      parts.push(`--tracker_params '${JSON.stringify(trackerParams)}'`);
    }

    if (state.display) parts.push("--display");
    if (!state.showBoxes) parts.push("--show_boxes false");
    if (state.showMasks) parts.push("--show_masks");
    if (state.showConfidence) parts.push("--show_confidence");
    if (state.showLabels) parts.push("--show_labels");
    if (!state.showIds) parts.push("--show_ids false");
    if (state.showTrajectories) parts.push("--show_trajectories");

    const outputValue = state.output.trim();
    if (outputValue) {
      parts.push(`--output ${outputValue}`);
      if (state.overwrite) {
        parts.push("--overwrite");
      }
    }

    return parts.join(" \\\n    ");
  }

  // Get validation errors
  function getValidationErrors(state) {
    const errors = [];

    if (state.showModelOptions) {
      if (state.confidence && !isValidDecimal01(state.confidence, 0.05)) {
        errors.push("Confidence must be between 0.05 and 1");
      }
      if (state.classes && !isValidClasses(state.classes)) {
        errors.push("Classes must contain only names, numbers, commas, and spaces");
      }
    }

    if (state.showTrackerOptions) {
      if (state.lostTrackBuffer && !isValidPositiveInt(state.lostTrackBuffer)) {
        errors.push("lost_track_buffer must be a positive integer");
      }
      if (state.trackActivationThreshold && !isValidDecimal01(state.trackActivationThreshold, 0.05)) {
        errors.push("track_activation_threshold must be between 0.05 and 1");
      }
      if (state.minimumConsecutiveFrames && !isValidPositiveInt(state.minimumConsecutiveFrames)) {
        errors.push("minimum_consecutive_frames must be a positive integer");
      }
      if (state.minimumIouThreshold && !isValidDecimal01(state.minimumIouThreshold, 0.05)) {
        errors.push("minimum_iou_threshold must be between 0.05 and 1");
      }
    }

    return errors;
  }

  const builder = new framework.TerminalCommandBuilder({
    root,
    defaults,
    numericFields,
    initialState,
    ids: {
      command: "cb-command",
      errors: "cb-errors",
      overwrite: "cb-overwrite",
    },
    generateCommand,
    getValidationErrors,
    inputSanitizers: {
      classes(value) {
        return value.replace(/[^\w,\s]/g, "");
      },
    },
    onInputChange({ key, value, isCommit, builder: instance }) {
      if (key !== "output") return;

      const hasOutput = value.trim() !== "";
      instance.setElementVisibleById(instance.ids.overwrite, hasOutput);

      if (isCommit && !hasOutput) {
        instance.setCheckboxField("overwrite", false, { refresh: false });
      }
    },
    renderBody(code, api) {
      const {
        ids,
        state,
        numericFields: numberConfig,
        createHeader,
        createSelector,
        createCheckbox,
        createTextInputRow,
        createNumericInputRow,
        createSelectorRow,
      } = api;

      // MODEL Section
      code.appendChild(createHeader("Model"));

      const modelTypeGrid = document.createElement("div");
      modelTypeGrid.className = "cb-model-type-grid cb-option-indent";

      const detectionSelector = createSelector(
        "detection",
        "detection",
        "modelType",
        state.modelType === "detection"
      );
      detectionSelector.style.gridColumn = "1";
      detectionSelector.classList.add("cb-selector--nudge-left");

      const segmentationSelector = createSelector(
        "segmentation",
        "segmentation",
        "modelType",
        state.modelType === "segmentation"
      );
      segmentationSelector.style.gridColumn = "3";

      modelTypeGrid.appendChild(detectionSelector);
      modelTypeGrid.appendChild(segmentationSelector);
      code.appendChild(modelTypeGrid);

      const modelSizeGrid = document.createElement("div");
      modelSizeGrid.className = "cb-model-size-grid cb-option-indent cb-model-size-grid--before-toggle";
      ["nano", "small", "medium", "large"].forEach((size) => {
        modelSizeGrid.appendChild(
          createSelector(size, size, "modelSize", state.modelSize === size)
        );
      });
      code.appendChild(modelSizeGrid);

      // Model Options
      const modelOptionsToggle = createCheckbox(
        "options",
        "showModelOptions",
        state.showModelOptions
      );
      modelOptionsToggle.classList.add("cb-option-indent", "cb-option-toggle");
      code.appendChild(modelOptionsToggle);

      const modelOptionsContent = document.createElement("div");
      modelOptionsContent.className = "cb-options cb-option-indent";
      modelOptionsContent.dataset.collapsibleFor = "showModelOptions";
      modelOptionsContent.style.display = state.showModelOptions ? "block" : "none";
      modelOptionsContent.appendChild(
        createNumericInputRow(
          "confidence",
          "confidence",
          state.confidence,
          numberConfig.confidence,
          "cb-row--anchor-3"
        )
      );
      modelOptionsContent.appendChild(
        createSelectorRow(
          "device",
          ["auto", "cpu", "cuda", "mps"],
          "device",
          state.device,
          "cb-row--anchor-3"
        )
      );
      modelOptionsContent.appendChild(
        createTextInputRow("classes", "person, car", "classes", state.classes, "cb-row--anchor-3")
      );
      code.appendChild(modelOptionsContent);

      // TRACKER Section
      code.appendChild(createHeader("Tracker"));

      const trackerSelectors = document.createElement("div");
      trackerSelectors.className =
        "cb-tracker-type-grid cb-option-indent cb-selectors-row--before-toggle";
      const bytetrackSelector = createSelector(
        "bytetrack",
        "bytetrack",
        "tracker",
        state.tracker === "bytetrack"
      );
      bytetrackSelector.style.gridColumn = "1";
      const sortSelector = createSelector("sort", "sort", "tracker", state.tracker === "sort");
      sortSelector.style.gridColumn = "3";
      trackerSelectors.appendChild(bytetrackSelector);
      trackerSelectors.appendChild(sortSelector);
      code.appendChild(trackerSelectors);

      // Tracker Options
      const trackerOptionsToggle = createCheckbox(
        "options",
        "showTrackerOptions",
        state.showTrackerOptions
      );
      trackerOptionsToggle.classList.add("cb-option-indent", "cb-option-toggle");
      code.appendChild(trackerOptionsToggle);

      const trackerOptionsContent = document.createElement("div");
      trackerOptionsContent.className = "cb-options cb-option-indent";
      trackerOptionsContent.dataset.collapsibleFor = "showTrackerOptions";
      trackerOptionsContent.style.display = state.showTrackerOptions ? "block" : "none";
      trackerOptionsContent.appendChild(
        createNumericInputRow(
          "lost_track_buffer",
          "lostTrackBuffer",
          state.lostTrackBuffer,
          numberConfig.lostTrackBuffer,
          "cb-row--anchor-3"
        )
      );
      trackerOptionsContent.appendChild(
        createNumericInputRow(
          "track_activation_threshold",
          "trackActivationThreshold",
          state.trackActivationThreshold,
          numberConfig.trackActivationThreshold,
          "cb-row--anchor-3"
        )
      );
      trackerOptionsContent.appendChild(
        createNumericInputRow(
          "minimum_consecutive_frames",
          "minimumConsecutiveFrames",
          state.minimumConsecutiveFrames,
          numberConfig.minimumConsecutiveFrames,
          "cb-row--anchor-3"
        )
      );
      trackerOptionsContent.appendChild(
        createNumericInputRow(
          "minimum_iou_threshold",
          "minimumIouThreshold",
          state.minimumIouThreshold,
          numberConfig.minimumIouThreshold,
          "cb-row--anchor-3"
        )
      );
      code.appendChild(trackerOptionsContent);

      // VISUALIZATION Section
      code.appendChild(createHeader("Visualization"));

      const vizCheckboxes = document.createElement("div");
      vizCheckboxes.className = "cb-checkboxes cb-option-indent";

      const vizOptions = [
        { key: "display", label: "display", checked: state.display },
        { key: "showBoxes", label: "boxes", checked: state.showBoxes },
        { key: "showMasks", label: "masks", checked: state.showMasks },
        { key: "showConfidence", label: "confidence", checked: state.showConfidence },
        { key: "showLabels", label: "labels", checked: state.showLabels },
        { key: "showIds", label: "ids", checked: state.showIds },
        { key: "showTrajectories", label: "trajectories", checked: state.showTrajectories },
      ];

      vizOptions.forEach((option) => {
        vizCheckboxes.appendChild(createCheckbox(option.label, option.key, option.checked));
      });
      code.appendChild(vizCheckboxes);

      // SOURCE Section
      code.appendChild(createHeader("Source"));
      code.appendChild(
        createTextInputRow(
          "path",
          "source.mp4",
          "source",
          state.source,
          "cb-option-indent cb-row--anchor-3"
        )
      );

      // OUTPUT Section
      code.appendChild(createHeader("Output"));
      code.appendChild(
        createTextInputRow(
          "path",
          "output.mp4",
          "output",
          state.output,
          "cb-option-indent cb-row--anchor-3"
        )
      );

      const overwriteWrapper = document.createElement("div");
      overwriteWrapper.id = ids.overwrite;
      overwriteWrapper.className = "cb-option-indent";
      overwriteWrapper.style.display = state.output.trim() ? "block" : "none";
      overwriteWrapper.appendChild(createCheckbox("overwrite", "overwrite", state.overwrite));
      code.appendChild(overwriteWrapper);
    },
  });

  builder.init();
});

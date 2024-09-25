import { useState } from "react";
import {
  Box,
  Button,
  ButtonGroup,
  InputAdornment,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { Link } from "react-router-dom";
import { CollapsibleBox, TitledBox } from "@/components";
import { FrequencyRangeList } from "./FrequencyRange";
import { useSettingsStore } from "@/stores";
import { filterObjectByKeys } from "@/utils/functions";

const formatKey = (key) => {
  // console.log(key);
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Wrapper components for each type
const BooleanField = ({ value, onChange }) => (
  <Switch checked={value} onChange={(e) => onChange(e.target.checked)} />
);

const StringField = ({ value, onChange, label }) => (
  <TextField value={value} onChange={onChange} label={label} />
);

const NumberField = ({ value, onChange, label }) => {
  const handleChange = (event) => {
    const newValue = event.target.value;
    // Only allow numbers and decimal point
    if (newValue === "" || /^\d*\.?\d*$/.test(newValue)) {
      onChange(newValue);
    }
  };

  return (
    <TextField
      type="text" // Using "text" instead of "number" for more control
      value={value}
      onChange={handleChange}
      label={label}
      // InputProps={{
      //   endAdornment: (
      //     <InputAdornment position="end">
      //       <span style={{ lineHeight: 1, display: "inline-block" }}>Hz</span>
      //     </InputAdornment>
      //   ),
      // }}
      inputProps={{
        pattern: "[0-9]*",
      }}
    />
  );
};

// Map component types to their respective wrappers
const componentRegistry = {
  boolean: BooleanField,
  string: StringField,
  number: NumberField,
};

const SettingsField = ({ path, Component, label, value, onChange, depth }) => {
  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      sx={{
        pl: depth * 2,
      }}
    >
      <Typography variant="body2">{label}</Typography>
      <Component
        value={value}
        onChange={(newValue) => onChange(path, newValue)}
        label={label}
      />
    </Stack>
  );
};

const SettingsSection = ({
  settings,
  title = null,
  path = [],
  onChange,
  depth = 0,
}) => {
  const boxTitle = title ? title : formatKey(path[path.length - 1]);

  // If we receive a primitive value, we need to render a component
  if (typeof settings !== "object") {
    const Component = componentRegistry[typeof settings];
    if (!Component) {
      console.error(`Invalid component type: ${typeof settings}`);
      return null;
    }
    return (
      <SettingsField
        Component={Component}
        label={boxTitle}
        value={settings}
        onChange={onChange}
        depth={depth + 1}
      />
      // <Stack direction="row">
      //   <Typography variant="body2">{boxTitle}</Typography>
      //   <Component label={title} value={settings} onChange={onChange} />
      // </Stack>
    );
  }

  // If we receive a nested object, we iterate over it and render recursively
  return (
    <TitledBox title={boxTitle} depth={depth} sx={{ borderRadius: 3 }}>
      {Object.entries(settings).map(([key, value]) => {
        if (key === "__field_type__") return null;

        const newPath = [...path, key];
        const label = key;
        const isPydanticModel =
          typeof value === "object" && "__field_type__" in value;

        const fieldType = isPydanticModel ? value.__field_type__ : typeof value;

        const Component = componentRegistry[fieldType];

        if (Component) {
          return (
            <SettingsField
              key={`${key}_settingsField`}
              path={newPath}
              Component={Component}
              label={formatKey(label)}
              value={value}
              onChange={onChange}
              depth={depth + 1}
            />
          );
        } else {
          return (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={value}
              path={newPath}
              onChange={onChange}
              depth={depth + 1}
            />
          );
        }
      })}
    </TitledBox>
  );
};

const SettingsContent = () => {
  const [selectedFeature, setSelectedFeature] = useState("");
  const settings = useSettingsStore((state) => state.settings);
  const updateSettings = useSettingsStore((state) => state.updateSettings);
  const frequencyRangeOrder = useSettingsStore(
    (state) => state.frequencyRangeOrder
  );
  const updateFrequencyRangeOrder = useSettingsStore(
    (state) => state.updateFrequencyRangeOrder
  );

  if (!settings) {
    return <div>Loading settings...</div>;
  }

  console.log(settings);

  const handleChange = (path, value) => {
    updateSettings((settings) => {
      let current = settings;
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
    });
  };

  const featureSettingsKeys = Object.keys(settings.features)
    .filter((feature) => settings.features[feature])
    .map((feature) => `${feature}_settings`);

  const enabledFeatures = filterObjectByKeys(settings, featureSettingsKeys);

  const preprocessingSettingsKeys = [
    "preprocessing",
    "raw_resampling_settings",
    "raw_normalization_settings",
    "preprocessing_filter",
  ];

  const postprocessingSettingsKeys = [
    "postprocessing",
    "feature_normalization_settings",
    "project_cortex_settings",
    "project_subcortex_settings",
  ];

  const generalSettingsKeys = [
    "sampling_rate_features_hz",
    "segment_length_features_ms",
  ];

  return (
    <Stack
      direction="row"
      alignItems="flex-start"
      justifyContent="flex-start"
      width="fit-content"
      gap={2}
      p={2}
    >
      <Stack sx={{ minWidth: "33%" }}>
        <TitledBox title="General Settings" depth={0}>
          {generalSettingsKeys.map((key) => (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={settings[key]}
              path={[key]}
              onChange={handleChange}
              depth={0}
            />
          ))}
        </TitledBox>

        <TitledBox title="Frequency Ranges" depth={0}>
          <FrequencyRangeList
            ranges={settings.frequency_ranges_hz}
            rangeOrder={frequencyRangeOrder}
            onOrderChange={updateFrequencyRangeOrder}
            onChange={handleChange}
          />
        </TitledBox>
      </Stack>

      <TitledBox
        title="Preprocessing Settings"
        depth={0}
        sx={{ borderRadius: 3 }}
      >
        {preprocessingSettingsKeys.map((key) => (
          <SettingsSection
            key={`${key}_settingsSection`}
            settings={settings[key]}
            path={[key]}
            onChange={handleChange}
            depth={0}
          />
        ))}
      </TitledBox>

      <TitledBox
        title="Postprocessing Settings"
        depth={0}
        sx={{ borderRadius: 3 }}
      >
        {postprocessingSettingsKeys.map((key) => (
          <SettingsSection
            key={`${key}_settingsSection`}
            settings={settings[key]}
            path={[key]}
            onChange={handleChange}
            depth={0}
          />
        ))}
      </TitledBox>

      <TitledBox title="Feature Settings" depth={0}>
        <Stack direction="row" gap={2}>
          <Box alignSelf={"flex-start"}>
            <SettingsSection
              settings={settings.features}
              path={["features"]}
              onChange={handleChange}
              depth={0}
              sx={{ alignSelf: "flex-start" }}
            />
          </Box>
          <Stack alignSelf={"flex-start"}>
            {Object.entries(enabledFeatures).map(
              ([feature, featureSettings]) => (
                <CollapsibleBox
                  key={`${feature}_collapsibleBox`}
                  title={formatKey(feature)}
                  defaultExpanded={false}
                >
                  <SettingsSection
                    key={`${feature}_settingsSection`}
                    settings={featureSettings}
                    path={[feature]}
                    onChange={handleChange}
                    depth={0}
                  />
                </CollapsibleBox>
              )
            )}
          </Stack>
        </Stack>
      </TitledBox>
    </Stack>
  );
};

export const Settings = () => {
  return (
    <Stack justifyContent="center" pb={2}>
      <SettingsContent />
      <Stack
        direction="row"
        width="fit-content"
        sx={{ position: "absolute", bottom: "2.5rem", right: "1rem", gap: 1 }}
        backgroundColor="background.level3"
        borderRadius={2}
        border="1px solid"
        borderColor={"divider"}
        p={1}
      >
        <Button variant="contained" color="primary" to="/decoding">
          Reset Settings
        </Button>
        <Button
          variant="contained"
          component={Link}
          color="primary"
          to="/decoding"
        >
          Run Stream
        </Button>
      </Stack>
    </Stack>
  );
};

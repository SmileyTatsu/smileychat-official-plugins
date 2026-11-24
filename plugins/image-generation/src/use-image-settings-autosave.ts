import type { SmileyPluginApi } from "../types/smileychat-plugin-api";
import {
    arePluginSettingsEqual,
    usePluginSettingsAutosave,
    type RequestState,
} from "./use-plugin-settings-autosave";

import { splitMasterPrompt } from "./master-prompt";
import { IMAGE_SETTINGS_KEY, type ImageGenerationSettings } from "./settings";

export { arePluginSettingsEqual as areImageSettingsEqual };
export type { RequestState };

export type UseImageSettingsAutosaveOptions = {
    api: SmileyPluginApi;
    settings: ImageGenerationSettings;
    onSettingsChange?: (settings: ImageGenerationSettings) => void;
};

export function useImageSettingsAutosave({
    api,
    settings,
    onSettingsChange,
}: UseImageSettingsAutosaveOptions) {
    return usePluginSettingsAutosave<ImageGenerationSettings>({
        api,
        settings,
        onSettingsChange,
        validate: (s) => {
            splitMasterPrompt(s.masterPrompt);
        },
    });
}

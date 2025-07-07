import { GoogleGenAI, GenerateContentResponse, File as GeminiFile, UploadFileConfig, UsageMetadata, Content, Part } from "@google/genai";
import { GeminiService, ChatHistoryItem, ThoughtSupportingPart, ModelOption, AppSettings } from '../types';
import { logService } from "./logService";

const POLLING_INTERVAL_MS = 2000; // 2 seconds
const MAX_POLLING_DURATION_MS = 10 * 60 * 1000; // 10 minutes

const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const result = reader.result as string;
            const base64Data = result.split(',')[1];
            if (base64Data) {
                resolve(base64Data);
            } else {
                reject(new Error("Failed to extract base64 data from file."));
            }
        };
        reader.onerror = error => reject(error);
    });
};

class GeminiServiceImpl implements GeminiService {
    constructor() {
        logService.info("GeminiService created.");
    }
    
    // MODIFIED: This function now accepts appSettings to handle the proxy URL.
    private _getClient(apiKey: string, appSettings?: AppSettings): GoogleGenAI {
      try {
          // If a custom proxy URL is provided, create a custom transport layer.
          // This will redirect all API calls from the official Google URL to your proxy.
          if (appSettings?.useCustomApiConfig && appSettings.apiProxyUrl) {
              logService.info(`Using custom API proxy: ${appSettings.apiProxyUrl}`);
              
              // We need to remove any trailing '/'.
              const cleanProxyUrl = appSettings.apiProxyUrl.replace(/\/+$/, '');

              const customTransport = {
                  fetch: (url: string, init?: RequestInit): Promise<Response> => {
                      // The official SDK builds URLs starting with this prefix.
                      const originalPrefix = 'https://generativelanguage.googleapis.com';
                      // Replace the official prefix with your proxy URL.
                      const newUrl = url.replace(originalPrefix, cleanProxyUrl);
                      
                      logService.debug(`Redirecting fetch from ${url} to ${newUrl}`);
                      return fetch(newUrl, init);
                  }
              };
              // Initialize the client with the custom transport.
              return new GoogleGenAI({ apiKey, transport: customTransport });
          }
          
          // If no proxy is set, use the default behavior.
          return new GoogleGenAI({ apiKey });

      } catch (error) {
          logService.error("Failed to initialize GoogleGenAI client:", error);
          throw error;
      }
    }

    // MODIFIED: This function now passes appSettings along.
    private _getApiClientOrThrow(apiKey?: string | null, appSettings?: AppSettings): GoogleGenAI {
        if (!apiKey) {
            const silentError = new Error("API key is not configured in settings or provided.");
            silentError.name = "SilentError";
            throw silentError;
        }
        return this._getClient(apiKey, appSettings);
    }

    private _buildGenerationConfig(
        modelId: string,
        systemInstruction: string,
        config: { temperature?: number; topP?: number },
        showThoughts: boolean,
        thinkingBudget: number
    ): any {
        const generationConfig: any = {
            ...config,
            systemInstruction: systemInstruction || undefined,
        };
        if (!generationConfig.systemInstruction) {
            delete generationConfig.systemInstruction;
        }
    
        const modelSupportsThinking = [
            'gemini-2.5-flash-lite-preview-06-17',
            'gemini-2.5-pro',
            'gemini-2.5-flash'
        ].includes(modelId);
    
        if (modelSupportsThinking) {
            if (showThoughts) {
                generationConfig.thinkingConfig = {
                    thinkingBudget: thinkingBudget,
                    includeThoughts: true,
                };
            } else {
                generationConfig.thinkingConfig = { thinkingBudget: 0 };
            }
        }
        
        return generationConfig;
    }

    // MODIFIED: Added appSettings parameter
    async getAvailableModels(apiKeysString: string | null, appSettings?: AppSettings): Promise<ModelOption[]> {
        logService.info('Fetching available models...');
        const keys = (apiKeysString || '').split('\n').map(k => k.trim()).filter(Boolean);

        if (keys.length === 0) {
            logService.warn('getAvailableModels called with no API keys.');
            throw new Error("API client not initialized. Configure API Key in settings.");
        }
        
        const randomKey = keys[Math.floor(Math.random() * keys.length)];
        const ai = this._getClient(randomKey, appSettings); // Pass appSettings here

        try {
          const modelPager = await ai.models.list(); 
          const availableModels: ModelOption[] = [];
          for await (const model of modelPager) {
             const supported = model.supportedActions;
             if (!supported || supported.includes('generateContent') || supported.includes('generateImages')) {
                availableModels.push({
                    id: model.name, 
                    name: model.displayName || model.name.split('/').pop() || model.name,
                    isPinned: false, 
                });
             }
          }

          if (availableModels.length > 0) {
            logService.info(`Fetched ${availableModels.length} models successfully.`);
            return availableModels.sort((a,b) => a.name.localeCompare(b.name));
          } else {
             logService.warn("API returned an empty list of models.");
             throw new Error("API returned an empty list of models.");
          }
        } catch (error) {
          logService.error("Failed to fetch available models from Gemini API:", error);
          throw error;
        }
    }

    // MODIFIED: Added appSettings parameter
    async uploadFile(apiKey: string, file: File, mimeType: string, displayName: string, signal: AbortSignal, appSettings?: AppSettings): Promise<GeminiFile> {
        logService.info(`Uploading file: ${displayName}`, { mimeType, size: file.size });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        if (signal.aborted) {
            logService.warn(`Upload for "${displayName}" cancelled before starting.`);
            const abortError = new Error("Upload cancelled by user.");
            abortError.name = "AbortError";
            throw abortError;
        }

        try {
            const uploadConfig: UploadFileConfig = { mimeType, displayName: encodeURIComponent(displayName) };
            
            let uploadedFile = await ai.files.upload({
                file: file,
                config: uploadConfig,
            });

            const startTime = Date.now();
            while (uploadedFile.state === 'PROCESSING' && (Date.now() - startTime) < MAX_POLLING_DURATION_MS) {
                if (signal.aborted) {
                    const abortError = new Error("Upload polling cancelled by user.");
                    abortError.name = "AbortError";
                    throw abortError;
                }
                logService.debug(`File "${displayName}" is PROCESSING. Polling again in ${POLLING_INTERVAL_MS / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, POLLING_INTERVAL_MS));
                
                if (signal.aborted) {
                     const abortError = new Error("Upload polling cancelled by user after timeout.");
                     abortError.name = "AbortError";
                     throw abortError;
                }

                try {
                    uploadedFile = await ai.files.get({ name: uploadedFile.name });
                } catch (pollError) {
                    logService.error(`Error polling for file status "${displayName}":`, pollError);
                    throw new Error(`Polling failed for file ${displayName}. Original error: ${pollError instanceof Error ? pollError.message : String(pollError)}`);
                }
            }

            if (uploadedFile.state === 'PROCESSING') {
                logService.warn(`File "${displayName}" is still PROCESSING after ${MAX_POLLING_DURATION_MS / 1000}s. Returning current state.`);
            }
            
            return uploadedFile;
        } catch (error) {
            logService.error(`Failed to upload and process file "${displayName}" to Gemini API:`, error);
            throw error;
        }
    }
    
    // MODIFIED: Added appSettings parameter
    async getFileMetadata(apiKey: string, fileApiName: string, appSettings?: AppSettings): Promise<GeminiFile | null> {
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        if (!fileApiName || !fileApiName.startsWith('files/')) {
            logService.error(`Invalid fileApiName format: ${fileApiName}. Must start with "files/".`);
            throw new Error('Invalid file ID format. Expected "files/your_file_id".');
        }
        try {
            logService.info(`Fetching metadata for file: ${fileApiName}`);
            const file = await ai.files.get({ name: fileApiName });
            return file;
        } catch (error) {
            logService.error(`Failed to get metadata for file "${fileApiName}" from Gemini API:`, error);
            if (error instanceof Error && (error.message.includes('NOT_FOUND') || error.message.includes('404'))) {
                return null;
            }
            throw error;
        }
    }

    // MODIFIED: Added appSettings parameter
    async generateImages(apiKey: string, modelId: string, prompt: string, aspectRatio: string, abortSignal: AbortSignal, appSettings?: AppSettings): Promise<string[]> {
        logService.info(`Generating image with model ${modelId}`, { prompt, aspectRatio });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        if (!prompt.trim()) {
            throw new Error("Image generation prompt cannot be empty.");
        }

        if (abortSignal.aborted) {
            const abortError = new Error("Image generation cancelled by user before starting.");
            abortError.name = "AbortError";
            throw abortError;
        }

        try {
            const response = await ai.models.generateImages({
                model: modelId,
                prompt: prompt,
                config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: aspectRatio },
            });

            if (abortSignal.aborted) {
                const abortError = new Error("Image generation cancelled by user.");
                abortError.name = "AbortError";
                throw abortError;
            }

            const images = response.generatedImages?.map(img => img.image.imageBytes) ?? [];
            if (images.length === 0) {
                throw new Error("No images generated. The prompt may have been blocked or the model failed to respond.");
            }
            
            return images;

        } catch (error) {
            logService.error(`Failed to generate images with model ${modelId}:`, error);
            throw error;
        }
    }

    // MODIFIED: Added appSettings parameter
    async generateSpeech(apiKey: string, modelId: string, text: string, voice: string, abortSignal: AbortSignal, appSettings?: AppSettings): Promise<string> {
        logService.info(`Generating speech with model ${modelId}`, { textLength: text.length, voice });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        if (!text.trim()) {
            throw new Error("TTS input text cannot be empty.");
        }

        try {
            const response = await ai.models.generateContent({
                model: modelId,
                contents: text,
                config: {
                    responseModalities: ['AUDIO'],
                    speechConfig: {
                        voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } },
                    },
                },
            });

            if (abortSignal.aborted) {
                const abortError = new Error("Speech generation cancelled by user.");
                abortError.name = "AbortError";
                throw abortError;
            }

            const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

            if (typeof audioData === 'string' && audioData.length > 0) {
                return audioData;
            }
            
            logService.error("TTS response did not contain expected audio data structure:", { response });
            const textError = response.text;
            if (textError) {
                throw new Error(`TTS generation failed: ${textError}`);
            }

            throw new Error('No audio data found in TTS response.');
        } catch (error) {
            logService.error(`Failed to generate speech with model ${modelId}:`, error);
            throw error;
        }
    }

    // MODIFIED: Added appSettings parameter
    async transcribeAudio(apiKey: string, audioFile: File, modelId: string, isThinkingEnabled: boolean, appSettings?: AppSettings): Promise<string> {
        logService.info(`Transcribing audio with model ${modelId}`, { fileName: audioFile.name, size: audioFile.size, thinking: isThinkingEnabled });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here

        const audioBase64 = await fileToBase64(audioFile);

        const audioPart: Part = { inlineData: { mimeType: audioFile.type, data: audioBase64, } };
        const textPart: Part = { text: "将此音频转录为文本。只返回转录的文本，不要回答音频中的问题。", };
        
        const config = {
          systemInstruction: "你是一个乐于助人的助手，负责逐字转录提供的音频文件，不得有任何遗漏或修改。",
          thinkingConfig: { thinkingBudget: isThinkingEnabled ? -1 : 0, },
        };

        try {
            const response = await ai.models.generateContent({
                model: modelId,
                contents: { parts: [textPart, audioPart] },
                config,
            });

            if (response.text) {
                return response.text;
            } else {
                const safetyFeedback = response.candidates?.[0]?.finishReason;
                if (safetyFeedback && safetyFeedback !== 'STOP') {
                     throw new Error(`Transcription failed due to safety settings: ${safetyFeedback}`);
                }
                throw new Error("Transcription failed. The model returned an empty response.");
            }
        } catch (error) {
            logService.error("Error during audio transcription:", error);
            throw error;
        }
    }

    // MODIFIED: Added appSettings parameter
    async sendMessageStream(
        apiKey: string, modelId: string, historyWithLastPrompt: ChatHistoryItem[],
        systemInstruction: string, config: { temperature?: number; topP?: number },
        showThoughts: boolean, thinkingBudget: number, abortSignal: AbortSignal,
        onChunk: (chunk: string) => void, onThoughtChunk: (chunk: string) => void,
        onError: (error: Error) => void, onComplete: (usageMetadata?: UsageMetadata) => void,
        appSettings?: AppSettings
    ): Promise<void> {
        logService.info(`Sending message to ${modelId} (stream)`, { hasSystemInstruction: !!systemInstruction, config, showThoughts, thinkingBudget });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        const generationConfig = this._buildGenerationConfig(modelId, systemInstruction, config, showThoughts, thinkingBudget);
        let finalUsageMetadata: UsageMetadata | undefined = undefined;

        try {
            const result = await ai.models.generateContentStream({ 
                model: modelId,
                contents: historyWithLastPrompt as Content[],
                config: generationConfig
            });

            for await (const chunkResponse of result) {
                if (abortSignal.aborted) {
                    logService.warn("Streaming aborted by signal.");
                    break;
                }
                if (chunkResponse.usageMetadata) {
                    finalUsageMetadata = chunkResponse.usageMetadata;
                }
                if (chunkResponse.candidates && chunkResponse.candidates[0]?.content?.parts?.length > 0) {
                    for (const part of chunkResponse.candidates[0].content.parts) {
                        if ('text' in part && typeof part.text === 'string' && part.text.length > 0) {
                            const pAsThoughtSupporting = part as ThoughtSupportingPart;
                            if (pAsThoughtSupporting.thought) {
                                onThoughtChunk(part.text);
                            } else {
                                onChunk(part.text);
                            }
                        }
                    }
                } else if (typeof chunkResponse.text === 'string' && chunkResponse.text.length > 0) {
                   onChunk(chunkResponse.text);
                }
            }
        } catch (error) {
            logService.error("Error sending message to Gemini (stream):", error);
            onError(error instanceof Error ? error : new Error(String(error) || "Unknown error during streaming."));
        } finally {
            logService.info("Streaming complete.", { usage: finalUsageMetadata });
            onComplete(finalUsageMetadata);
        }
    }

    // MODIFIED: Added appSettings parameter
    async sendMessageNonStream(
        apiKey: string, modelId: string, historyWithLastPrompt: ChatHistoryItem[],
        systemInstruction: string, config: { temperature?: number; topP?: number },
        showThoughts: boolean, thinkingBudget: number, abortSignal: AbortSignal,
        onError: (error: Error) => void,
        onComplete: (fullText: string, thoughtsText?: string, usageMetadata?: UsageMetadata) => void,
        appSettings?: AppSettings
    ): Promise<void> {
        logService.info(`Sending message to ${modelId} (non-stream)`, { hasSystemInstruction: !!systemInstruction, config, showThoughts, thinkingBudget });
        const ai = this._getApiClientOrThrow(apiKey, appSettings); // Pass appSettings here
        const generationConfig = this._buildGenerationConfig(modelId, systemInstruction, config, showThoughts, thinkingBudget);
        
        try {
            if (abortSignal.aborted) {
                onComplete("", "", undefined);
                return;
            }
            const response: GenerateContentResponse = await ai.models.generateContent({ 
                model: modelId,
                contents: historyWithLastPrompt as Content[],
                config: generationConfig
            });
            if (abortSignal.aborted) {
                onComplete("", "", undefined);
                return;
            }
            let fullText = "";
            let thoughtsText = "";
            if (response.candidates && response.candidates[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if ('text' in part && typeof part.text === 'string' && part.text.length > 0) {
                        const pAsThoughtSupporting = part as ThoughtSupportingPart;
                        if (pAsThoughtSupporting.thought) {
                            thoughtsText += part.text;
                        } else {
                            fullText += part.text;
                        }
                    }
                }
            }
            if (!fullText && response.text) {
                fullText = response.text;
            }
            logService.info("Non-stream call complete.", { usage: response.usageMetadata });
            onComplete(fullText, thoughtsText || undefined, response.usageMetadata);
        } catch (error) {
            logService.error("Error sending message to Gemini (non-stream):", error);
            onError(error instanceof Error ? error : new Error(String(error) || "Unknown error during non-streaming call."));
        }
    }
}

export const geminiServiceInstance: GeminiService = new GeminiServiceImpl();

"use client";

import { useState, useRef, useCallback } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

type ProgressEvent = { message: string; percent: number };
type CompleteEvent = { complete: true; summary: any; type: "image" | "video" };
type ErrorEvent = { error: string };
type StreamEvent = ProgressEvent | CompleteEvent | ErrorEvent;

function isProgress(e: StreamEvent): e is ProgressEvent {
  return "message" in e && !("complete" in e) && !("error" in e);
}
function isComplete(e: StreamEvent): e is CompleteEvent {
  return "complete" in e;
}
function isError(e: StreamEvent): e is ErrorEvent {
  return "error" in e;
}

// Convert absolute output path to a backend-served URL
function toStaticUrl(absolutePath: string): string {
  // Paths look like /home/.../output/uploads/<uuid>/... or output/...
  // Backend serves /output/ → OUTPUT_DIR
  const idx = absolutePath.indexOf("output/");
  if (idx >= 0) {
    return `${BACKEND_URL}/output/${absolutePath.slice(idx + "output/".length)}`;
  }
  return `${BACKEND_URL}/output/${absolutePath.split("/").pop()}`;
}

const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);
const DownloadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);

function UploadForm({ onUpload }: { onUpload: (file: File) => void }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFile) onUpload(selectedFile);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setSelectedFile(f);
  };

  const isVideo = selectedFile && selectedFile.type.startsWith("video/");

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
          dragActive ? "border-blue-500 bg-blue-50" : "border-slate-300 hover:border-slate-400"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragActive(false);
          const f = e.dataTransfer.files[0];
          if (f) setSelectedFile(f);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          type="file"
          ref={inputRef}
          onChange={handleFileChange}
          accept="image/*,video/*"
          className="hidden"
        />
        <UploadIcon />
        <p className="mt-2 text-sm text-slate-600">
          {selectedFile
            ? `${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB)`
            : "Click or drag an image or video file here"}
        </p>
        {selectedFile && (
          <p className="mt-1 text-xs text-slate-500">
            Type: {isVideo ? "Video → 9:16 reel + 4 stills" : "Image → 4 aspect variants"}
          </p>
        )}
      </div>
      <button
        type="submit"
        disabled={!selectedFile}
        className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium disabled:opacity-50 hover:bg-blue-700 transition"
      >
        Process with ReframeAI
      </button>
    </form>
  );
}

function ProgressView({ events }: { events: ProgressEvent[] }) {
  const latest = events[events.length - 1];
  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-slate-700">Progress</span>
        <span className="text-sm text-slate-500">{latest?.percent ?? 0}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2.5 mb-4">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${latest?.percent ?? 0}%` }}
        />
      </div>
      <div className="space-y-2 max-h-48 overflow-y-auto">
        {events.map((e, i) => (
          <div key={i} className="flex items-center gap-3 text-sm">
            <div className={`w-2 h-2 rounded-full ${i === events.length - 1 ? "bg-blue-600" : "bg-slate-300"}`} />
            <span className="text-slate-700">{e.message}</span>
            <span className="text-slate-500 ml-auto">{e.percent}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImageResults({ summary }: { summary: any }) {
  const variants = summary.variants || {};
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {Object.entries(variants).map(([ratio, v]: [string, any]) => {
        const url = toStaticUrl(v.render_path);
        return (
          <div key={ratio} className="bg-white rounded-lg shadow p-4">
            <div className="bg-slate-100 rounded overflow-hidden mb-3">
              <img src={url} alt={`Crop ${ratio}`} className="w-full h-48 object-cover" />
            </div>
            <div className="flex justify-between items-center">
              <span className="font-medium text-sm">{ratio}</span>
              <a href={url} download className="inline-flex items-center gap-1 text-xs bg-slate-100 px-2 py-1 rounded hover:bg-slate-200">
                <DownloadIcon /> Download
              </a>
            </div>
            <p className={`text-xs mt-1 ${v.passed ? "text-green-600" : "text-red-600"}`}>
              {v.passed ? "✓ Passed" : "✗ Failed"}
            </p>
          </div>
        );
      })}
    </div>
  );
}

function VideoResults({ summary }: { summary: any }) {
  const outputs = summary.outputs || {};
  const stills: string[] = outputs.stills || [];
  return (
    <div className="space-y-6">
      {/* Reel */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-medium mb-2">Vertical Reel (9:16)</h3>
        <div className="aspect-video max-w-xs mx-auto">
          <video src={toStaticUrl(outputs.reel)} controls className="w-full h-auto rounded" />
        </div>
        <div className="flex justify-center mt-2">
          <a href={toStaticUrl(outputs.reel)} download className="inline-flex items-center gap-1 text-sm bg-slate-100 px-3 py-1 rounded hover:bg-slate-200">
            <DownloadIcon /> Download Reel
          </a>
        </div>
      </div>

      {/* Stills in 4 aspect ratios */}
      {stills.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {stills.map((still) => {
            // Extract ratio from filename (e.g. "..._16_9.jpg" → "16:9")
            const basename = still.split("/").pop() || "";
            const match = basename.match(/_(\d+)_(\d+)\./);
            const ratioLabel = match ? `${match[1]}:${match[2]}` : "Still";
            return (
              <div key={still} className="bg-white rounded-lg shadow p-3 text-center">
                <div className="relative mb-2">
                  <img src={toStaticUrl(still)} alt={`Still ${ratioLabel}`} className="w-full h-32 object-cover rounded" />
                  <span className="absolute top-1 right-1 bg-slate-800/70 text-white text-xs px-1.5 py-0.5 rounded">
                    {ratioLabel}
                  </span>
                </div>
                <a href={toStaticUrl(still)} download className="inline-flex items-center gap-1 text-xs bg-slate-100 px-2 py-1 rounded hover:bg-slate-200">
                  <DownloadIcon /> Download
                </a>
              </div>
            );
          })}
        </div>
      )}

      {/* Debug video */}
      {outputs.debug && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-medium mb-2">Debug Overlay (tracking + speaker)</h3>
          <video src={toStaticUrl(outputs.debug)} controls className="w-full max-w-2xl rounded" />
        </div>
      )}
    </div>
  );
}

function ResultView({ summary, type }: { summary: any; type: "image" | "video" }) {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Results</h2>
      {type === "image" ? <ImageResults summary={summary} /> : <VideoResults summary={summary} />}

      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="font-medium mb-2">Validation</h3>
        <p className={summary.validation_passed ? "text-green-600" : "text-red-600"}>
          {summary.validation_passed ? "✓ All checks passed" : "✗ Validation issues found"}
        </p>
        {summary.validation_errors?.length > 0 && (
          <ul className="text-sm text-red-600 mt-1">
            {summary.validation_errors.map((e: string, i: number) => <li key={i}>• {e}</li>)}
          </ul>
        )}
        <div className="mt-2 text-xs text-slate-500">
          {summary.timing && `Total: ${summary.timing.total}s`}
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [step, setStep] = useState<"upload" | "processing" | "done">("upload");
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [assetType, setAssetType] = useState<"image" | "video">("image");
  const [error, setError] = useState<string | null>(null);

  const handleUpload = useCallback(async (file: File) => {
    setStep("processing");
    setError(null);
    setEvents([]);
    setSummary(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`${BACKEND_URL}/process`, {
        method: "POST",
        body: formData,
        headers: { "Accept": "text/event-stream" },
      });

      if (!resp.ok) throw new Error(`Server error: ${resp.status} ${resp.statusText}`);

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        // Parse SSE format: data: {...}\n\n
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6).trim();
            try {
              const evt: StreamEvent = JSON.parse(jsonStr);
              if (isProgress(evt)) {
                setEvents((prev) => [...prev, evt]);
              } else if (isComplete(evt)) {
                setSummary(evt.summary);
                setAssetType(evt.type);
                setStep("done");
              } else if (isError(evt)) {
                setError(evt.error);
                setStep("upload");
              }
            } catch {
              // Not JSON, skip
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
      setStep("upload");
    }
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">ReframeAI</h1>
          <p className="text-slate-600">Subject-aware media cropping for social platforms. Upload an image or video.</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          {step === "upload" && (
            <>
              <h2 className="text-xl font-semibold mb-4">Upload Media</h2>
              <UploadForm onUpload={handleUpload} />
              {error && <p className="mt-4 text-red-600 text-sm">{error}</p>}
            </>
          )}

          {step === "processing" && (
            <>
              <h2 className="text-xl font-semibold mb-6 text-center">Processing your media…</h2>
              <ProgressView events={events} />
            </>
          )}

          {step === "done" && summary && <ResultView summary={summary} type={assetType} />}
        </div>

        <div className="text-center">
          <button
            onClick={() => {
              setStep("upload");
              setEvents([]);
              setSummary(null);
              setError(null);
            }}
            className="text-blue-600 hover:underline text-sm font-medium"
          >
            ← Process another file
          </button>
        </div>
      </div>
    </div>
  );
}

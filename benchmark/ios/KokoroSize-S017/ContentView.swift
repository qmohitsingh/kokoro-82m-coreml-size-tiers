import AVFoundation
import CryptoKit
import SwiftUI

// Production screen: type a Kokoro phoneme string and speak it offline, and the
// iPhone 12 speed test on the frozen cases. The Listen configuration adds only
// the hidden listening screen.
struct ContentView: View {
    @State private var phonemes = "həlˈoʊ wˈɜːld."
    @State private var status = "Ready."
    @State private var busy = false
    private let player = AudioPlayer()

    var body: some View {
        NavigationStack {
            List {
                Section("Speak") {
                    TextField("Kokoro phonemes", text: $phonemes, axis: .vertical)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                    Text("Kokoro phonemes (IPA). Example = \"hello world\".")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button("Speak") { speak() }.disabled(busy || phonemes.isEmpty)
                }
                Section("Speed") {
                    Text("Two throwaway runs, then 10 timed runs. The score is the middle RTF. Pass is 0.30 or lower on every length.")
                    Button("Run Core ML speed test") { runCoreMLSpeed() }.disabled(busy)
                }
                #if S017_LISTEN
                Section {
                    NavigationLink("Hidden listening (S017)") { HiddenListenView() }
                }
                #endif
                Section("Result") {
                    Text(status).font(.footnote).textSelection(.enabled)
                }
            }
            .navigationTitle("Kokoro")
            .task {
                // Device gate: `devicectl device process launch ... -- -coreml-speed`
                if CommandLine.arguments.contains("-coreml-speed") {
                    runCoreMLSpeed()
                }
                // Same-audio check between configurations: `... -- -hash`
                if CommandLine.arguments.contains("-hash") {
                    writeHash()
                }
                #if S017_LISTEN
                // `devicectl device process launch ... -- -make-listen`
                if CommandLine.arguments.contains("-make-listen") {
                    UIApplication.shared.isIdleTimerDisabled = true
                    HiddenListenMaker.start(
                        tick: { text in Task { @MainActor in status = "Making S017 listening clips " + text } },
                        finish: { text in
                            Task { @MainActor in
                                status = text
                                UIApplication.shared.isIdleTimerDisabled = false
                            }
                        }
                    )
                }
                #endif
            }
        }
    }

    private func speak() {
        let (ids, unknown) = Phonemes.tokens(phonemes)
        guard !ids.isEmpty, ids.count <= Phonemes.maxPhonemes else {
            status = "Enter 1 to \(Phonemes.maxPhonemes) Kokoro phonemes."
            return
        }
        busy = true
        status = unknown.isEmpty ? "Synthesizing..." : "Skipped \(unknown.joined(separator: " ")). Synthesizing..."
        let note = status
        Task.detached {
            do {
                let start = Date()
                let audio = try SpeechEngine.shared.speak(ids: ids)
                let wall = Date().timeIntervalSince(start)
                await MainActor.run {
                    player.play(samples: audio)
                    status = note.replacingOccurrences(of: "Synthesizing...", with: "")
                        + String(format: "%.2fs of speech in %.2fs.", Double(audio.count) / 24000, wall)
                    busy = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    busy = false
                }
            }
        }
    }

    private func runCoreMLSpeed() {
        busy = true
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Running Core ML speed test..."
        Task.detached {
            do {
                let rows = try Bench.coreMLSpeed(try Bench.load()) { line in
                    Task { @MainActor in status += "\n" + line }
                }
                let lines = rows.map { "\($0.id) Core ML median RTF \(String(format: "%.3f", $0.medianRTF))" }
                await MainActor.run {
                    status = (lines + ["Saved coreml-speed.json in the app Files folder."]).joined(separator: "\n")
                    busy = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    busy = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        }
    }

    // SHA-256 of the fp32 samples for the frozen 3s case -> Documents/hash-<build>.json.
    private func writeHash() {
        Task.detached {
            do {
                guard let chunk = try Bench.load().cases.first(where: { $0.id == "3s" })?.chunks.first else {
                    throw BenchError.missing("3s case")
                }
                let audio = try chunk.synthesize(CoreMLChain(), VoiceTable())
                let digest = audio.withUnsafeBytes { SHA256.hash(data: $0) }
                let hex = digest.map { String(format: "%02x", $0) }.joined()
                try Bench.write(["build": AppBuild.name, "case": "3s", "samples": "\(audio.count)", "sha256": hex],
                                name: "hash-\(AppBuild.name).json")
                await MainActor.run { status = "3s \(audio.count) samples sha256 \(hex)" }
            } catch {
                await MainActor.run { status = String(describing: error) }
            }
        }
    }
}

// One loaded chain and voice table for typed input, made on first use.
final class SpeechEngine: @unchecked Sendable {
    static let shared = SpeechEngine()
    private let lock = NSLock()
    private var chain: CoreMLChain?
    private var voice: VoiceTable?

    func speak(ids: [Int64]) throws -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        if chain == nil { chain = try CoreMLChain() }
        if voice == nil { voice = try VoiceTable() }
        return try chain!.synthesize(ids: [0] + ids + [0], style: voice!.style(phonemeCount: ids.count))
    }
}

final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        try? AVAudioSession.sharedInstance().setActive(true)
        try? engine.start()
    }

    func play(samples: [Float]) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData?[0].update(from: src.baseAddress!, count: samples.count)
        }
        node.stop()
        node.scheduleBuffer(buffer)
        node.play()
    }
}

import AVFoundation
import SwiftUI

struct ContentView: View {
    @State private var status = "Load a test, then run it on the iPhone 12."
    @State private var busy = false
    @State private var loadingModel = false
    @State private var loadSeconds = 0
    @State private var looping = false
    @State private var stop = StopFlag()
    @State private var file: BenchFile?
    @State private var made: [String: [Float]] = [:]
    @State private var coreMLAudio: [Float] = []
    @State private var pairs: [ListenPair] = []
    private let player = AudioPlayer()

    var body: some View {
        NavigationStack {
            List {
                Section("Speed") {
                    Text("Two throwaway runs, then 10 timed runs. The score is the middle RTF. Pass is 0.30 or lower on every length.")
                    Button("Run Core ML speed test") { runCoreMLSpeed() }.disabled(busy)
                }
                Section("Memory") {
                    Text("Recorded, not scored. One clip at each length, then three 30-second loops. The speed test does not run this sampler.")
                    Button("Run memory diagnostic") { runMemory() }.disabled(busy)
                }
                Section("Core ML probe") {
                    Text("Loads the five converted 3-second models, makes one clip, and checks its sample count.")
                    Button("Run 3s Core ML") { runCoreMLProbe() }.disabled(busy)
                    Button("Play Core ML result") {
                        player.play(samples: coreMLAudio)
                    }
                    .disabled(coreMLAudio.isEmpty || busy)
                }
                #if S014
                Section {
                    NavigationLink("Hidden listening (S014)") { HiddenListenView() }
                }
                #endif
                Section("Listen") {
                    Text("A and B are unlabeled. One is the saved Kokoro reference. The other is Core ML. Mark the one you prefer, or tie.")
                    Button("Make Core ML listening pairs") { prepareListen() }.disabled(busy)
                    ForEach(pairs) { pair in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(pair.caseId)
                            // Borderless: in a List row, plain buttons all fire on one tap.
                            HStack {
                                Button("Play A") { play(pair, side: .a) }
                                Button("Play B") { play(pair, side: .b) }
                            }
                            .buttonStyle(.borderless)
                            HStack {
                                Button("Prefer A") { mark(pair, "A") }
                                Button("Prefer B") { mark(pair, "B") }
                                Button("Tie") { mark(pair, "tie") }
                            }
                            .buttonStyle(.borderless)
                            if let choice = pair.choice {
                                Text("Marked \(choice)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Section("60 minutes") {
                    Text("Repeats the 30-second Core ML clip until an hour has passed. Leave the app open and in front. A forced close is a fail.")
                    Button("Start 60 minute Core ML loop") { runStability() }.disabled(busy)
                    Button("Stop loop") { stop.stop() }.disabled(!looping)
                }
                Section("Result") {
                    if loadingModel {
                        HStack {
                            ProgressView()
                            Text("Loading model... \(loadSeconds)s")
                        }
                    }
                    Text(status).font(.footnote).textSelection(.enabled)
                }
            }
            .navigationTitle("Kokoro bench")
            .task {
                load()
                // Device gate: `devicectl device process launch ... -- -coreml-speed`
                if CommandLine.arguments.contains("-coreml-speed") {
                    runCoreMLSpeed()
                }
                if CommandLine.arguments.contains("-memory") {
                    runMemory()
                }
                #if S014
                // `devicectl device process launch ... -- -make-listen`
                if CommandLine.arguments.contains("-make-listen") {
                    UIApplication.shared.isIdleTimerDisabled = true
                    HiddenListenMaker.start(
                        tick: { text in Task { @MainActor in status = "Making S014 listening clips " + text } },
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

    private func load() {
        do {
            file = try Bench.load()
            status = "Ready. \(file?.cases.map(\.id).joined(separator: ", ") ?? "")"
        } catch {
            status = String(describing: error)
        }
    }

    private func runCoreMLProbe() {
        guard let item = file?.cases.first(where: { $0.id == "3s" }),
              let chunk = item.chunks.first
        else { return }
        busy = true
        loadingModel = true
        loadSeconds = 0
        status = "Loading the 3s Core ML chain..."
        Task { @MainActor in
            while loadingModel {
                try? await Task.sleep(for: .seconds(1))
                if loadingModel { loadSeconds += 1 }
            }
        }
        Task.detached {
            do {
                let loadStart = Date()
                let chain = try CoreMLChain(tokenCount: 56, frameCount: 146)
                let loadWall = Date().timeIntervalSince(loadStart)
                await MainActor.run {
                    loadingModel = false
                    status = String(format: "Core ML loaded in %.1fs. Synthesizing...", loadWall)
                }
                let start = Date()
                let audio = try chain.synthesize(chunk)
                let wall = Date().timeIntervalSince(start)
                let audioSeconds = Double(audio.count) / 24_000
                #if S014
                // synthesize already checked audio.count against the student's own frames.
                let matched = audio.count.isMultiple(of: 600)
                #else
                let matched = audio.count == chunk.samples
                #endif
                let report = CoreMLProbeReport(
                    id: item.id,
                    samples: audio.count,
                    expectedSamples: chunk.samples,
                    sampleCountMatched: matched,
                    loadSeconds: loadWall,
                    wallSeconds: wall,
                    audioSeconds: audioSeconds,
                    rtf: wall / audioSeconds,
                    thermalState: Bench.thermal()
                )
                try Bench.write(report, name: "coreml.json")
                await MainActor.run {
                    coreMLAudio = audio
                    status = String(
                        format: "3s Core ML complete.\n%d samples\n%.2fs wall\nRTF %.3f\nThermal: %@\nSaved coreml.json in the app Files folder.",
                        report.samples,
                        report.wallSeconds,
                        report.rtf,
                        report.thermalState
                    )
                    busy = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    loadingModel = false
                    busy = false
                }
            }
        }
    }

    private func runMemory() {
        guard let file else { return }
        busy = true
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Running memory diagnostic..."
        Task.detached {
            do {
                let report = try MemoryDiagnostic.run(file) { line in
                    Task { @MainActor in status += "\n" + line }
                }
                try Bench.write(report, name: "memory.json")
                let lines = report.parts.map {
                    "\($0.id) part \($0.part): load \(MemoryDiagnostic.megabytes($0.peakLoadBytes)), synthesis \(MemoryDiagnostic.megabytes($0.peakSynthesisBytes))"
                }
                await MainActor.run {
                    status = (lines + ["Saved memory.json in the app Files folder."]).joined(separator: "\n")
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

    private func runCoreMLSpeed() {
        guard let file else { return }
        busy = true
        loadingModel = true
        loadSeconds = 0
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Running Core ML speed test..."
        Task { @MainActor in
            while loadingModel {
                try? await Task.sleep(for: .seconds(1))
                if loadingModel { loadSeconds += 1 }
            }
        }
        Task.detached {
            do {
                let rows = try Bench.coreMLSpeed(file) { line in
                    Task { @MainActor in
                        if line.contains("models loaded") { loadingModel = false }
                        status += "\n" + line
                    }
                }
                try Bench.write(rows, name: "coreml-speed.json")
                let lines = rows.map {
                    "\($0.id) Core ML median RTF \(String(format: "%.3f", $0.medianRTF))"
                }
                await MainActor.run {
                    status = (
                        lines + ["Saved coreml-speed.json in the app Files folder."]
                    ).joined(separator: "\n")
                    busy = false
                    loadingModel = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    busy = false
                    loadingModel = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        }
    }

    private func prepareListen() {
        guard let file else { return }
        busy = true
        loadingModel = true
        loadSeconds = 0
        status = "Making Core ML audio for the listening test..."
        Task { @MainActor in
            while loadingModel {
                try? await Task.sleep(for: .seconds(1))
                if loadingModel { loadSeconds += 1 }
            }
        }
        Task.detached {
            do {
                var audio: [String: [Float]] = [:]
                for item in file.cases {
                    var samples: [Float] = []
                    for (index, chunk) in item.chunks.enumerated() {
                        guard chunk.samples.isMultiple(of: 600) else { throw BenchError.badOutput }
                        await MainActor.run {
                            status = "Core ML listen: \(item.id) part \(index + 1). Loading..."
                        }
                        let chain = try CoreMLChain(
                            tokenCount: chunk.inputIds.count,
                            frameCount: chunk.samples / 600
                        )
                        await MainActor.run {
                            loadingModel = false
                            status = "Core ML listen: \(item.id) part \(index + 1). Synthesizing..."
                        }
                        let part = try autoreleasepool { try chain.synthesize(chunk) }
                        samples.append(contentsOf: part)
                    }
                    audio[item.id] = samples
                }
                let next = file.cases.map {
                    ListenPair(
                        caseId: $0.id,
                        build: AppBuild.name,
                        candidate: "coreml",
                        aIsReference: Bool.random()
                    )
                }
                // Same bundle id keeps Documents across installs. Replace the
                // previous listen.json before any mark, so an old score cannot
                // be read as this build.
                let madeAudio = audio
                try Bench.write(next, name: "listen.json")
                await MainActor.run {
                    made = madeAudio
                    pairs = next
                    status = "\(AppBuild.name) pairs are ready. Play A and B, then mark one. The screen does not say which side is the reference."
                    busy = false
                    loadingModel = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    busy = false
                    loadingModel = false
                }
            }
        }
    }

    private func play(_ pair: ListenPair, side: ListenPair.Side) {
        let wantReference = (side == .a) == pair.aIsReference
        if wantReference {
            guard let item = file?.cases.first(where: { $0.id == pair.caseId }) else { return }
            player.playWavs(item.wavs)
        } else if let samples = made[pair.caseId] {
            player.play(samples: samples)
        }
    }

    private func mark(_ pair: ListenPair, _ choice: String) {
        guard let index = pairs.firstIndex(where: { $0.id == pair.id }) else { return }
        pairs[index].choice = choice
        let marks = pairs.compactMap { item -> String? in
            guard let choice = item.choice else { return nil }
            return "\(item.caseId) marked \(choice)"
        }
        status = marks.joined(separator: "\n") + "\nSaved listen.json."
        try? Bench.write(pairs, name: "listen.json")
    }

    private func runStability() {
        guard let file else { return }
        stop = StopFlag()
        let flag = stop
        busy = true
        looping = true
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Stability loop running."
        Task.detached {
            do {
                try Bench.stability(file, stop: flag) { note in
                    Task { @MainActor in status = note }
                }
                await MainActor.run {
                    status = "Stability file saved. Finished means the hour completed."
                    busy = false
                    looping = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            } catch {
                await MainActor.run {
                    status = String(describing: error)
                    busy = false
                    looping = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        }
    }
}

struct CoreMLProbeReport: Encodable {
    let id: String
    let samples: Int
    let expectedSamples: Int
    let sampleCountMatched: Bool
    let loadSeconds: Double
    let wallSeconds: Double
    let audioSeconds: Double
    let rtf: Double
    let thermalState: String
}

struct ListenPair: Identifiable, Encodable {
    let id = UUID()
    let caseId: String
    let build: String
    let candidate: String
    let aIsReference: Bool
    var choice: String?
    enum Side { case a, b }
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
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData?[0].update(from: src.baseAddress!, count: samples.count)
        }
        node.stop()
        node.scheduleBuffer(buffer)
        node.play()
    }

    func playWavs(_ names: [String]) {
        var samples: [Float] = []
        for name in names {
            let stem = (name as NSString).deletingPathExtension
            guard let url = Bundle.main.url(forResource: stem, withExtension: "wav"),
                  let file = try? AVAudioFile(forReading: url),
                  let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
            else { continue }
            try? file.read(into: buffer)
            if let channel = buffer.int16ChannelData?[0] {
                for i in 0..<Int(buffer.frameLength) {
                    samples.append(Float(channel[i]) / 32768)
                }
            } else if let channel = buffer.floatChannelData?[0] {
                samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
            }
        }
        play(samples: samples)
    }
}

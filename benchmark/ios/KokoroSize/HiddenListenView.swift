#if S014
import AVFoundation
import SwiftUI

// Blind listening test for the S014 candidate. Clips live in the app's Documents,
// copied there with devicectl, never in the app bundle:
//   listen-s014/sentences.json        ids, token ids, style rows (made on the Mac)
//   listen-s014/ref/<id>.wav          original Kokoro, made on the Mac
//   listen-s014/student/<id>.wav      student arm, made on the Mac
//   listen-out-m8/s014/<id>.wav          S014 Core ML chain, made here on the phone
//   listen-out-m8/s014/frames.json       frames and checks per clip
//   listen-out-m8/make-status.json, make-error.txt
//   listen-s014-m8.json                  one row per vote
// The screen never shows an arm name or a sentence id.

struct ListenSentence: Decodable {
    let id: String
    let inputIds: [Int64]
    let style: [Float]

    enum CodingKeys: String, CodingKey {
        case id
        case inputIds = "input_ids"
        case style
    }
}

struct ListenClipInfo: Codable {
    var frames: Int
    var samples: Int
    var peak: Double
    var rmsDb: Double
    var nonFinite: Int

    enum CodingKeys: String, CodingKey {
        case frames, samples, peak
        case rmsDb = "rms_db"
        case nonFinite = "non_finite"
    }
}

struct ListenResult: Codable {
    let listener: String
    let listenerIndex: Int
    let pairIndex: Int
    let sentenceId: String
    let arm: String
    let aIsReference: Bool
    let choice: String
    let mosA: Int
    let mosB: Int
    let timestamp: String
}

struct ListenPairPlan {
    let sentenceId: String
    let arm: String
    let aIsReference: Bool
}

enum ListenFiles {
    static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    static let dir = documents.appendingPathComponent("listen-s014")
    // -m8: the mixed8 generator. The int4 pilot stays in listen-out and listen-s014.json.
    static let results = documents.appendingPathComponent("listen-s014-m8.json")
    // Inputs are copied into listen-s014 by devicectl, which the app cannot write to.
    // Everything the app makes goes into listen-out-m8, a folder the app creates itself.
    static let out = documents.appendingPathComponent("listen-out-m8")

    static func clip(_ folder: String, _ id: String) -> URL {
        (folder == "s014" ? out : dir).appendingPathComponent(folder).appendingPathComponent("\(id).wav")
    }
}

// SplitMix64, so a listener's session is the same every time it is built.
struct ListenRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    mutating func below(_ count: Int) -> Int { Int(next() % UInt64(count)) }

    mutating func shuffle<T>(_ items: inout [T]) {
        guard items.count > 1 else { return }
        for index in stride(from: items.count - 1, to: 0, by: -1) {
            items.swapAt(index, below(index + 1))
        }
    }
}

enum ListenPlan {
    // GOAL.md sound gate: one blind listener, the four lengths plus a few unseen sentences.
    static let pairsPerListener = 8
    // The int4 pilot used entries 0-23; start after them so the sentences are unseen.
    static let firstEntry = 24

    // One fixed shuffled order of all sentences (the file is grouped by category).
    // Listener k takes 8 consecutive entries starting at 24 + 8k, wrapping around. Every pair is original vs S014. The
    // reference sits on side A in exactly half the pairs, in an order seeded by k.
    static func session(listenerIndex: Int, sentenceIds: [String]) -> [ListenPairPlan] {
        var order = sentenceIds
        var sentenceRNG = ListenRNG(seed: 0x5014)
        sentenceRNG.shuffle(&order)
        let start = (firstEntry + listenerIndex * pairsPerListener) % order.count
        let picked = (0..<pairsPerListener).map { order[(start + $0) % order.count] }
        var sides = Array(repeating: true, count: pairsPerListener / 2)
            + Array(repeating: false, count: pairsPerListener / 2)
        var rng = ListenRNG(seed: 0xA5_0000 + UInt64(listenerIndex))
        rng.shuffle(&sides)
        return zip(picked, sides).map { id, aIsReference in
            ListenPairPlan(sentenceId: id, arm: "s014", aIsReference: aIsReference)
        }
    }
}

// Makes the S014 arm with the same CoreMLChain the speed test uses.
enum HiddenListenMaker {
    static func run(tick: @escaping (String) -> Void) throws {
        let sentences = try JSONDecoder().decode(
            [ListenSentence].self,
            from: Data(contentsOf: ListenFiles.dir.appendingPathComponent("sentences.json"))
        )
        let out = ListenFiles.out.appendingPathComponent("s014")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let infoURL = out.appendingPathComponent("frames.json")
        var infos = (try? JSONDecoder().decode(
            [String: ListenClipInfo].self, from: Data(contentsOf: infoURL)
        )) ?? [:]
        var chain: CoreMLChain?
        for (index, sentence) in sentences.enumerated() {
            let wav = ListenFiles.clip("s014", sentence.id)
            if FileManager.default.fileExists(atPath: wav.path), infos[sentence.id] != nil {
                tick("\(index + 1)/\(sentences.count)")
                continue
            }
            let count = sentence.inputIds.count
            if chain == nil {
                tick("Loading Core ML models... \(index)/\(sentences.count)")
                chain = try CoreMLChain(tokenCount: count, frameCount: 0)
            }
            let current = chain!.retargeted(tokenCount: count)
            let chunk = BenchChunk(inputIds: sentence.inputIds, style: sentence.style, samples: 0)
            let audio = try autoreleasepool { try current.synthesize(chunk) }
            var peak: Double = 0
            var squares: Double = 0
            var bad = 0
            var clean = audio
            for position in clean.indices {
                if !clean[position].isFinite {
                    bad += 1
                    clean[position] = 0
                }
                peak = max(peak, Double(abs(clean[position])))
                squares += Double(clean[position]) * Double(clean[position])
            }
            let rms = (squares / Double(max(1, clean.count))).squareRoot()
            try wavData(clean).write(to: wav, options: .atomic)
            infos[sentence.id] = ListenClipInfo(
                frames: audio.count / 600,
                samples: audio.count,
                peak: peak,
                rmsDb: 20 * log10(max(rms, 1e-9)),
                nonFinite: bad
            )
            try JSONEncoder().encode(infos).write(to: infoURL, options: .atomic)
            tick("\(index + 1)/\(sentences.count)")
        }
    }

    // Documents/listen-out-m8/make-status.json says how far it got, and why it stopped.
    static func start(tick: @escaping (String) -> Void, finish: @escaping (String) -> Void) {
        Task.detached {
            try? FileManager.default.createDirectory(at: ListenFiles.out, withIntermediateDirectories: true)
            let statusURL = ListenFiles.out.appendingPathComponent("make-status.json")
            func note(_ text: String, done: Bool, error: Bool) {
                let body: [String: Any] = ["text": text, "done": done, "error": error]
                if let data = try? JSONSerialization.data(withJSONObject: body) {
                    try? data.write(to: statusURL, options: .atomic)
                }
            }
            do {
                try run { text in
                    note(text, done: false, error: false)
                    tick(text)
                }
                note("done", done: true, error: false)
                finish("S014 clips ready.")
            } catch {
                let text = String(describing: error)
                note(text, done: true, error: true)
                try? text.write(
                    to: ListenFiles.out.appendingPathComponent("make-error.txt"),
                    atomically: true, encoding: .utf8
                )
                finish("Making clips failed: \(text)")
            }
        }
    }

    // 24 kHz, 16-bit, mono, no loudness change.
    static func wavData(_ samples: [Float]) -> Data {
        var data = Data()
        func u32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        data.append("RIFF".data(using: .ascii)!)
        u32(36 + bytes)
        data.append("WAVEfmt ".data(using: .ascii)!)
        u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16)
        data.append("data".data(using: .ascii)!)
        u32(bytes)
        for sample in samples {
            let scaled = (max(-1, min(1, sample)) * 32767).rounded()
            u16(UInt16(bitPattern: Int16(scaled)))
        }
        return data
    }
}

@MainActor
final class HiddenListenModel: ObservableObject {
    @Published var makeStatus = "Not started."
    @Published var making = false
    @Published var listener = ""
    @Published var message = ""
    @Published var session: [ListenPairPlan] = []
    @Published var pairIndex = 0
    @Published var finished = false
    @Published var choice: String?
    @Published var mosA: Int?
    @Published var mosB: Int?

    private var listenerName = ""
    private var listenerIndex = 0
    private var results: [ListenResult] = []
    private let player = AudioPlayer()

    var canNext: Bool { choice != nil && mosA != nil && mosB != nil }
    var started: Bool { !session.isEmpty }

    func makeClips() {
        making = true
        UIApplication.shared.isIdleTimerDisabled = true
        makeStatus = "Starting..."
        HiddenListenMaker.start(
            tick: { text in Task { @MainActor in self.makeStatus = text } },
            finish: { text in
                Task { @MainActor in
                    self.making = false
                    self.makeStatus = text
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        )
    }

    private func loadResults() -> [ListenResult]? {
        guard FileManager.default.fileExists(atPath: ListenFiles.results.path) else { return [] }
        return try? JSONDecoder().decode([ListenResult].self, from: Data(contentsOf: ListenFiles.results))
    }

    func startSession() {
        let name = listener.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { message = "Type a listener name first."; return }
        guard let existing = loadResults() else {
            message = "listen-s014.json cannot be read. Not starting, so no votes are overwritten."
            return
        }
        guard let data = try? Data(contentsOf: ListenFiles.dir.appendingPathComponent("sentences.json")),
              let sentences = try? JSONDecoder().decode([ListenSentence].self, from: data)
        else { message = "listen-s014/sentences.json is missing."; return }

        let mine = existing.filter { $0.listener.lowercased() == name.lowercased() }
        let names = Set(existing.map { $0.listener.lowercased() })
        let index = mine.first?.listenerIndex ?? names.count
        let plan = ListenPlan.session(listenerIndex: index, sentenceIds: sentences.map(\.id))

        let missing = plan.filter { pair in
            !FileManager.default.fileExists(atPath: ListenFiles.clip("ref", pair.sentenceId).path)
                || !FileManager.default.fileExists(atPath: ListenFiles.clip(pair.arm, pair.sentenceId).path)
        }.count
        guard missing == 0 else {
            message = "\(missing) clips are missing. Copy the Mac clips and tap Make S014 clips first."
            return
        }
        let done = Set(mine.map(\.pairIndex))
        guard let next = plan.indices.first(where: { !done.contains($0) }) else {
            message = "This listener has finished. Use a new name."
            return
        }
        results = existing
        listenerName = mine.first?.listener ?? name
        listenerIndex = index
        session = plan
        pairIndex = next
        finished = false
        choice = nil; mosA = nil; mosB = nil
        message = ""
    }

    private func samples(side: String) -> [Float] {
        let pair = session[pairIndex]
        let wantReference = (side == "A") == pair.aIsReference
        let url = ListenFiles.clip(wantReference ? "ref" : pair.arm, pair.sentenceId)
        guard let file = try? AVAudioFile(forReading: url),
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: buffer)) != nil,
              let channel = buffer.floatChannelData?[0]
        else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    func play(_ side: String) {
        guard started, !finished else { return }
        let audio = samples(side: side)
        if audio.isEmpty { message = "Could not read that clip." } else { player.play(samples: audio) }
    }

    func next() {
        guard canNext, let choice, let mosA, let mosB else { return }
        let pair = session[pairIndex]
        results.append(ListenResult(
            listener: listenerName,
            listenerIndex: listenerIndex,
            pairIndex: pairIndex,
            sentenceId: pair.sentenceId,
            arm: pair.arm,
            aIsReference: pair.aIsReference,
            choice: choice,
            mosA: mosA,
            mosB: mosB,
            timestamp: ISO8601DateFormatter().string(from: Date())
        ))
        // Saved before moving on, so a killed app loses nothing already submitted.
        do {
            try JSONEncoder().encode(results).write(to: ListenFiles.results, options: .atomic)
            message = ""
        } catch {
            message = "Save failed: \(error)"
            results.removeLast()
            return
        }
        self.choice = nil; self.mosA = nil; self.mosB = nil
        if pairIndex + 1 >= session.count {
            finished = true
        } else {
            pairIndex += 1
        }
    }

    func reset() {
        session = []
        finished = false
        listener = ""
        message = ""
    }
}

struct HiddenListenView: View {
    @StateObject private var model = HiddenListenModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button("Make S014 clips") { model.makeClips() }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.making)
                Text(model.makeStatus).font(.footnote).foregroundStyle(.secondary)
                Divider()
                TextField("Listener name", text: $model.listener)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .disabled(model.started)
                if !model.started {
                    Button("Start session") { model.startSession() }.buttonStyle(.bordered)
                } else if model.finished {
                    Text("All \(model.session.count) pairs done. Thank you.").font(.headline)
                    Button("New listener") { model.reset() }.buttonStyle(.bordered)
                } else {
                    pairBody
                }
                if !model.message.isEmpty {
                    Text(model.message).font(.footnote).foregroundStyle(.red)
                }
            }
            .padding()
        }
        .navigationTitle("Hidden listening")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var pairBody: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Sentence \(model.pairIndex + 1)/\(model.session.count)").font(.headline)
            HStack {
                wide("Play A", selected: false) { model.play("A") }
                wide("Play B", selected: false) { model.play("B") }
            }
            HStack {
                wide("Prefer A", selected: model.choice == "A") { model.choice = "A" }
                wide("Prefer B", selected: model.choice == "B") { model.choice = "B" }
                wide("Tie", selected: model.choice == "tie") { model.choice = "tie" }
            }
            Text("A quality, 1 bad to 5 excellent").font(.footnote)
            HStack {
                ForEach(1...5, id: \.self) { score in
                    wide("\(score)", selected: model.mosA == score) { model.mosA = score }
                }
            }
            Text("B quality, 1 bad to 5 excellent").font(.footnote)
            HStack {
                ForEach(1...5, id: \.self) { score in
                    wide("\(score)", selected: model.mosB == score) { model.mosB = score }
                }
            }
            Button("Next") { model.next() }
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
                .disabled(!model.canNext)
        }
    }

    @ViewBuilder
    private func wide(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        if selected {
            Button(action: action) { Text(title).frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent)
        } else {
            Button(action: action) { Text(title).frame(maxWidth: .infinity) }
                .buttonStyle(.bordered)
        }
    }
}
#endif
